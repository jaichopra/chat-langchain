"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from pathlib import Path

from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def ingest_docs():
    """Get documents from web pages."""

    docs = []
    for p in Path("./pandas.documentation").rglob("*.html"):
        if p.is_dir():
            continue
        loader = UnstructuredHTMLLoader(p)
        raw_document = loader.load()
        docs.append(raw_document)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
