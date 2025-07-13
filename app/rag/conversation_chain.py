import warnings

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_conversational_chain(vectorstore, model_name, temperature=0.7, k=25):
    """
    Set up a Conversational Retrieval Chain using OpenAI LLM, a retriever over the vectorstore, and memory.

    Args:
        vectorstore (Chroma): The vectorstore containing embedded document chunks.
        model_name (str): Name of the OpenAI model to use (e.g., 'gpt-3.5-turbo').
        temperature (float): Sampling temperature for generation.
        k (int): Number of chunks to retrieve for context.

    Returns:
        ConversationalRetrievalChain: A configured conversational chain for RAG.
    """
    llm = ChatOpenAI(temperature=temperature, model_name=model_name)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
