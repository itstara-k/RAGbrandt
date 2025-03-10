'''
This script reads, tokenizes, and embeds text data from PDFs of public domain drawing guides and stores them in a Chroma vector database
so that the RAG model can retrieve relevant data more efficiently (ie. avoids processing all of the PDFs each time a user submits a query)
'''

import os
import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

logging.getLogger().setLevel(logging.ERROR)
loader = PyPDFDirectoryLoader(os.path.join(os.getcwd(), "data"))
data_on_pdf = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(data_on_pdf)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(documents=splits, embedding=embeddings_model, persist_directory=os.path.join(os.getcwd(), "vectordb"))
