"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from pathlib import Path
import time

from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores.lance_dataset import LanceDataset

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def ingest_docs():
    """Get documents from web pages."""

    docs_path = Path("docs.pkl")
    docs = []
    if not docs_path.exists():
        for p in Path("./pandas.documentation").rglob("*.html"):
            if p.is_dir():
                continue
            loader = UnstructuredHTMLLoader(p)
            raw_document = loader.load()
            docs = docs + raw_document

        with docs_path.open("wb") as fh:
            pickle.dump(docs, fh)
    else:
        with docs_path.open("rb") as fh:
            docs = pickle.load(fh)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = LanceDataset.from_documents(documents, embeddings, uri="pandas.lance")

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

if __name__ == "__main__":
    ingest_docs()
