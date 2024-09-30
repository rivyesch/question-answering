
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

import os
import tempfile
import uuid
import pandas as pd
import re
import boto3
from langchain_community.llms.bedrock import Bedrock
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain.vectorstores import FAISS
import streamlit as st
import re

def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).

    Parameters:
        filename (str): The filename to clean

    Returns:
        str: The cleaned filename
    """
    # Regular expression to find "(number)" pattern
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    return new_filename

def get_pdf_text(uploaded_file): 
    """
    Load a PDF document from an uploaded file and return it as a list of documents

    Parameters:
        uploaded_file (file-like object): The uploaded PDF file to load

    Returns:
        list: A list of documents created from the uploaded PDF file
    """
    try:
        # Read file content
        input_file = uploaded_file.read()

        # Create a temporary file (PyPDFLoader requires a file path to read the PDF,
        # it can't work directly with file-like objects or byte streams that we get from Streamlit's uploaded_file)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        # load PDF document
        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()

        return documents
    
    finally:
        # Ensure the temporary file is deleted when we're done with it
        os.unlink(temp_file.name)


def split_document(documents, chunk_size, chunk_overlap):    
    """
    Function to split generic text into smaller chunks.
    chunk_size: The desired maximum size of each chunk (default: 400)
    chunk_overlap: The number of characters to overlap between consecutive chunks (default: 20).

    Returns:
        list: A list of smaller text chunks created from the generic text
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap,
                                          length_function=len,
                                          separators=["\n\n", "\n", " "])
    
    return text_splitter.split_documents(documents)


# def get_embedding_function():
#     """
#     Return an OpenAIEmbeddings object, which is used to create vector embeddings from text.
#     The embeddings model used is "text-embedding-ada-002" and the OpenAI API key is provided
#     as an argument to the function.

#     Parameters:
#         api_key (str): The OpenAI API key to use when calling the OpenAI Embeddings API.

#     Returns:
#         OpenAIEmbeddings: An OpenAIEmbeddings object, which can be used to create vector embeddings from text.
#     """
    

#     return embeddings


def create_vectorstore_from_texts(documents):
    
    # Load Hugging Face model for embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Generate embeddings using Hugging Face model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Create FAISS vector store (RAG component)
    vectorstore = FAISS.from_documents(documents, embeddings)

    # # Save FAISS vector store to file for future use
    # vectorstoredb.save_local("faiss_store")
        
    return vectorstore


# def load_vectorstore(file_name, api_key, vectorstore_path="db"):

#     """
#     Load a previously saved Chroma vector store from disk.

#     :param file_name: The name of the file to load (without the path)
#     :param api_key: The OpenAI API key used to create the vector store
#     :param vectorstore_path: The path to the directory where the vector store was saved (default: "db")
    
#     :return: A Chroma vector store object
#     """
#     embedding_function = get_embedding_function(api_key)
#     return Chroma(persist_directory=vectorstore_path, 
#                   embedding_function=embedding_function, 
#                   collection_name=clean_filename(file_name))

# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}

Answer only the relevant portion of the context, don't include the template.
"""

# class AnswerWithSources(BaseModel):
#     """An answer to the question, with sources and reasoning."""
#     answer: str = Field(description="Answer to question")
#     sources: str = Field(description="Full direct text chunk from the context used to answer the question")
#     reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")
    

# class ExtractedInfoWithSources(BaseModel):
#     """Extracted information about the research article"""
#     paper_title: AnswerWithSources
#     paper_summary: AnswerWithSources
#     publication_year: AnswerWithSources
#     paper_authors: AnswerWithSources

def format_docs(docs):
    """
    Format a list of Document objects into a single string.

    :param docs: A list of Document objects

    :return: A string containing the text of all the documents joined by two newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)

# retriever | format_docs passes the question through the retriever, generating Document objects, and then to format_docs to generate strings;
# RunnablePassthrough() passes through the input question unchanged.
def query_document(vectorstore, query, access_key_id, secret_access_key):

    """
    Query a vector store with a question and return a structured response.

    :param vectorstore: A Chroma vector store object
    :param query: The question to ask the vector store
    :param api_key: The OpenAI API key to use when calling the OpenAI Embeddings API

    :return: A pandas DataFrame with three rows: 'answer', 'source', and 'reasoning'
    """

    boto_session = boto3.session.Session(
    aws_access_key_id = access_key_id,
    aws_secret_access_key = secret_access_key,
    region_name = 'ap-south-1'
    )

    # Load the Bedrock client using Boto3.
    bedrock = boto_session.client(service_name='bedrock-runtime')

    llm = Bedrock(
    model_id="meta.llama3-8b-instruct-v1:0", 
    client=bedrock, 
    model_kwargs={
        "max_gen_len": 512,  # Maximum generation length
        "temperature": 0.5,  # Temperature for controlling randomness
        "top_p": 0.9         # Nucleus sampling (top-p) parameter
        }
    )

    retriever=vectorstore.as_retriever(search_type="similarity")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            )

    structured_response = rag_chain.invoke(query)
    # df = pd.DataFrame([structured_response.dict()])

    # # Transforming into a table with two rows: 'answer' and 'source'
    # answer_row = []
    # source_row = []
    # reasoning_row = []

    # for col in df.columns:
    #     answer_row.append(df[col][0]['answer'])
    #     source_row.append(df[col][0]['sources'])
    #     reasoning_row.append(df[col][0]['reasoning'])

    # # Create new dataframe with two rows: 'answer' and 'source'
    # structured_response_df = pd.DataFrame([answer_row, source_row, reasoning_row], columns=df.columns, index=['answer', 'source', 'reasoning'])
  
    return structured_response


def clean_text(text):
    """
    Clean up unwanted symbols or formatting from the text.
    
    Parameters:
    text (str): The raw text to clean.
    
    Returns:
    str: The cleaned text.
    """
    # Remove repeated pipes (|) or any sequences of pipe characters
    cleaned_text = re.sub(r'\|+', '', text)

    # Remove timestamps or sequences of numbers (like '2021-07-22 17:45:38' or '0.000')
    cleaned_text = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}|\d+\.\d+', '', cleaned_text)

    # Remove repeated zeroes or sequences of numeric data (e.g., '0.000 0.000')
    cleaned_text = re.sub(r'0\.000\s+', '', cleaned_text)

    # Remove any excessive spaces or newlines
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Optionally, remove any leading/trailing whitespace and redundant newlines
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text
