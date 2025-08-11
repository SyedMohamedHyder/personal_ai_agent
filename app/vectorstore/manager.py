import os

# RAG imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def create_vectorstore(chunks, persist_directory, overwrite=True):
    """
    Create a Chroma vector store from document chunks.

    Args:
        chunks (list): List of document chunks to embed and store.
        persist_directory (str): Directory path where the vector store will be saved.
        overwrite (bool): If True, deletes existing collection before creating a new one.

    Returns:
        Chroma: The created Chroma vector store instance.
    """
    embeddings = OpenAIEmbeddings()

    if overwrite and os.path.exists(persist_directory):
        Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_directory
    )

    return vectorstore


def get_collection(vectorstore):
    """
    Retrieve the underlying collection from a Chroma vector store.

    Args:
        vectorstore (Chroma): An instance of the Chroma vector store.

    Returns:
        Collection: The internal collection object.
    """
    return vectorstore._collection


def inspect_vectorstore(collection):
    """
    Inspect the contents of a Chroma vector store collection.

    Prints the number of vectors and their embedding dimensionality.

    Args:
        collection: The underlying Chroma collection object.
    """
    count = collection.count()

    sample = collection.get(limit=1, include=["embeddings"])
    embeddings = sample.get("embeddings", [])

    if len(embeddings) > 0:
        dimensions = len(embeddings[0])
        print(
            f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
        )
    else:
        print("No embeddings found in the vector store.")
