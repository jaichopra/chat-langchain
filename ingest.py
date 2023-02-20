"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from pathlib import Path
import time

from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import pdb

def ingest_docs():
    """Get documents from web pages."""

    docs = []
    for p in Path("./pandas.documentation").rglob("*.html"):        
        if p.is_dir():
            continue
        loader = UnstructuredHTMLLoader(p)
        raw_document = loader.load()
        docs = docs + raw_document

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = None
    counter = 0
    total_count = len(documents)

    for doc in documents:
        print(f"generating embeddings {counter} / {total_count}")
        texts = doc.page_content

        if vectorstore is None:
            vectorstore = FAISS.from_texts(texts, embeddings)
        else:
            vectorstore.add_texts(texts)
        
        counter += 1

        # We sleep to throttle OpenAI API limits
        time.sleep(5)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
