import warnings
warnings.filterwarnings("ignore")

import pickle
from simple_colors import *
from bs4 import BeautifulSoup
from IPython.display import Markdown
import functools, operator, requests, os, json
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain.retrievers import MergerRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, PyPDFLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import END, MessageGraph


def chunkByWord(text):
    return len(text.split(" "))


def get_web_retriever_tool(embeddings, urls: list, retriever_name: str, retriever_description: str, recursive=False, recursive_depth=5):
    """Create a retriever tool by taking the list of urls as input. Use RecursiveUrlLoader on demand"""
    
    if recursive:
        docs = [RecursiveUrlLoader(url=url, max_depth=recursive_depth, extractor=lambda x: BeautifulSoup(x, "html.parser").text).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
    else:
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=400)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, length_function = chunkByWord, chunk_overlap=50, add_start_index = False)
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever()
    retrieve_tool = create_retriever_tool(retriever=retriever, name=retriever_name, description=retriever_description)

    return retrieve_tool


def get_retriever_tool(llm, embeddings, file_path: str, retriever_name: str, retriever_description: str):
    """Create a retriever tool from the pre loaded vector store. Input is OpenAI embeddings, file_path, retriever name and discussion."""
    
#     loader = DirectoryLoader(file_path, glob="**/*.pdf", loader_cls = PyPDFLoader, use_multithreading=True, show_progress=True)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, length_function = chunkByWord, chunk_overlap=50, add_start_index = False)
#     doc_splits = text_splitter.split_documents(documents)

#     # save to disk
#     vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embeddings, persist_directory='vectorstore/annual_reports/')

    ## load from disk
    vectorstore = Chroma(persist_directory='vectorstore/annual_reports/', embedding_function=embeddings)

    # Define 2 diff retrievers with 2 diff search type.
    retriever_1 = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    retriever_2 = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    lotr = MergerRetriever(retrievers=[retriever_1, retriever_2])

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=lotr)
    retrieve_tool = create_retriever_tool(retriever=compression_retriever, name=retriever_name, description=retriever_description)
    return retrieve_tool



def get_pdf_retriever_tool(llm, embeddings, file_path: str, retriever_name: str, retriever_description: str):
    """Create a retriever tool by taking the pdf as input. Input is llm, OpenAI embeddings, file_path, retriever name and description"""
    
    loader = PyPDFLoader(file_path, extract_images=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, length_function = chunkByWord, chunk_overlap=50, add_start_index = False)
    doc_splits = text_splitter.split_documents(documents)

    # save to disk
    vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embeddings) #, persist_directory='vectorstore/temp/'

    ## load from disk
#     vectorstore = Chroma(persist_directory='vectorstore/temp/', embedding_function=embeddings)

    # Define 2 diff retrievers with 2 diff search type.
    retriever_1 = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    retriever_2 = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    lotr = MergerRetriever(retrievers=[retriever_1, retriever_2])

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=lotr)
    retrieve_tool = create_retriever_tool(retriever=compression_retriever, name=retriever_name, description=retriever_description)
    return retrieve_tool



def get_retriever_tool_from_retriever(file_path: str, retriever_name: str, retriever_description: str):
    """Create a retriever tool by taking the pre loaded retriever as input in a pickle format. Input is file path, retriever name and description."""
    
    with open(file_path, 'rb') as handle:
        compression_retriever = pickle.load(handle)

    retrieve_tool = create_retriever_tool(retriever=compression_retriever, name=retriever_name, description=retriever_description)
    return retrieve_tool


def get_supervisor_node(llm:ChatOpenAI, system_prompt, members):
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}] }},
            "required": ["next"],
        },
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),]).partial(options=str(options), members=", ".join(members))

    supervisor_node = (prompt | llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputFunctionsParser())
    return supervisor_node


def get_agent_node(llm:ChatOpenAI, name, tools, system_prompt):
  """Defines the agent with a name and a prompt. State of the LangGraph is passed across each node"""
  def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}
    # Create an agent using agent exector (langchain old style) 
    agent = create_agent(llm, tools, system_prompt)

    # Convert the agent into a node for the LG
    agent_node = functools.partial(agent_node, agent=agent, name=name)
    return agent_node


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    """Helper function for creating agent executor"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
        

def run_graph(graph, question, verbose=True):
    response = graph.invoke({
        "messages": [HumanMessage(content=question)]
    })
    display(Markdown(response['messages'][-1].content.replace('$',"`$`")))
#     stream = graph.stream({"messages": [HumanMessage(content=question)]})
#     display(Markdown(stream['__end__']['messages'][-1].content))
    
#         for s in graph.stream({"messages": [HumanMessage(content=question)]}):
#             if verbose:
#                 if "__end__" not in s:
#                     print(s)
#                     print("---------------------------------------------------------------------------------------------------------------------------------")



async def run_reflection(graph, question, verbose=True):
    async for event in graph.astream(
        [
            HumanMessage(
                content=question
            )
        ],
    ):
        if 'generate' in event.keys():
            print("-------------------------------- {} --------------------------------".format(red("Response", ['bold'])))
            display(Markdown(event['generate'].content.replace('$',"`$`")))
        elif 'reflect' in event.keys():
            print("-------------------------------- {} --------------------------------".format(red("Reflection", ['bold'])))
            display(Markdown(event['reflect'].content.replace('$',"`$`")))
        else:
            print("-------------------------------- {} --------------------------------".format(red("Final Answer", ['bold'])))
            display(Markdown(event['__end__'][-1].content.replace('$',"`$`")))
            
            
def get_reflection_agent(llm, reflection_prompt):
    reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", reflection_prompt,),
        MessagesPlaceholder(variable_name="messages"),
    ]
    )
    reflection_agent = reflection_prompt | llm
    return reflection_agent


def generate_answer(graph, question):
    response = graph.invoke({
        "messages": [HumanMessage(content=question)]
    })
    return response['output']

def reflect_answer(reflection_agent, question, response):
    request = HumanMessage(content=question)
    reflection = ""
    for chunk in reflection_agent.stream({"messages": [request, HumanMessage(content=response)]}):
        print(chunk.content, end="")
        reflection += chunk.content


def generation_node(state: Sequence[BaseMessage]):
    response = generate_agent.invoke({"messages": state})
    return AIMessage(content=response['output'])


async def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    # Other messages we need to adjust
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    res = await reflect.ainvoke({"messages": translated})
    # We treat the output of this as human feedback for the generator
    return HumanMessage(content=res.content)


