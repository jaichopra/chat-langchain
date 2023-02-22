"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from pathlib import Path
import time
import re

from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores.lance_dataset import LanceDataset

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def get_document_title(document):
    m = str(document.metadata["source"])
    title = re.findall("pandas.documentation(.*).html", m)
    if title[0] is not None:
        return(title[0])
    return ''

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
            m = {}
            m["title"] = get_document_title(raw_document[0])
            m["version"] = "2.0rc0"
            raw_document[0].metadata = raw_document[0].metadata | m
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
    LanceDataset.from_documents(documents, embeddings, uri="pandas.lance")

if __name__ == "__main__":
    ingest_docs()
