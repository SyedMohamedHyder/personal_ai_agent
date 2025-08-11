import os
import glob

# RAG imports
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader


def add_metadata(doc, doc_type):
    """
    Add a 'doc_type' field to the document's metadata.

    Args:
        doc (Document): The document to annotate.
        doc_type (str): The type or source folder of the document.

    Returns:
        Document: The updated document with added metadata.
    """
    doc.metadata["doc_type"] = doc_type
    return doc


def load_folder_documents(folder):
    """
    Load all markdown documents from a given folder and tag them with metadata.

    Args:
        folder (str): Path to the folder containing documents.

    Returns:
        list: List of documents with metadata added.
    """
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(
        folder,
        glob="**/*",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        recursive=True,
        show_progress=True,
    )
    folder_docs = loader.load()
    return [add_metadata(doc, doc_type) for doc in folder_docs]


def load_documents(folders):
    """
    Load documents from a list of folders.

    Args:
        folders (list): List of folder paths.

    Returns:
        list: Combined list of documents from all folders.
    """
    documents = []
    for folder in folders:
        documents.extend(load_folder_documents(folder))
    return documents


def load_documents_from_knowledge_base(knowledge_base: str):
    """
    Load all documents from the specified knowledge base directory.

    Uses the provided knowledge_base path (can include wildcards) to find all folders,
    then loads and tags markdown documents within those folders.

    Args:
        knowledge_base (str): Path to the knowledge base directory or pattern (e.g., "knowledge-base/*").

    Returns:
        list: A list of LangChain Document objects with added metadata.
    """
    folders = glob.glob(f"{knowledge_base}/*")
    return load_documents(folders)


def split_into_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller text chunks for processing.

    Args:
        documents (list): List of LangChain Document objects.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        list: List of chunked Document objects.
    """
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
