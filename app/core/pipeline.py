import os
from dotenv import load_dotenv

from vectorstore.manager import create_vectorstore, get_collection
from loaders.document import load_documents_from_knowledge_base, split_into_chunks


def setup_environment():
    """
    Load environment variables from a .env file and set the OpenAI API key.

    This function ensures that `OPENAI_API_KEY` is available in the environment,
    falling back to a default placeholder if not explicitly set.
    """
    load_dotenv(override=True)
    os.environ["OPENAI_API_KEY"] = os.getenv(
        "OPENAI_API_KEY", "your-key-if-not-using-env"
    )


def load_and_vectorize(knowledge_base, vector_db):
    """
    Load documents from the knowledge base, split them into chunks, and build a vector store.

    Args:
        knowledge_base (str): Path to the knowledge base directory.
        vector_db (str): Path to the persistent vector store directory.

    Returns:
        tuple:
            - documents (list): List of original loaded documents.
            - chunks (list): List of text chunks from the documents.
            - vectorstore (Chroma): The populated Chroma vector store instance.
            - collection (Collection): The underlying Chroma collection from the vector store.
    """
    documents = load_documents_from_knowledge_base(knowledge_base)
    chunks = split_into_chunks(documents)
    vectorstore = create_vectorstore(chunks, vector_db)
    collection = get_collection(vectorstore)

    return documents, chunks, vectorstore, collection
