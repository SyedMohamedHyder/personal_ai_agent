import warnings

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_conversational_chain(
    vectorstore, model_name, temperature=0.7, k=25, system_prompt=None
):
    """
    Set up a Conversational Retrieval Chain using OpenAI LLM, a retriever over the vectorstore, and memory.

    Args:
        vectorstore (Chroma): The vectorstore containing embedded document chunks.
        model_name (str): Name of the OpenAI model to use (e.g., 'gpt-3.5-turbo').
        temperature (float): Sampling temperature for generation.
        k (int): Number of chunks to retrieve for context.
        system_prompt (str, optional): System prompt to guide the AI's behavior and responses.

    Returns:
        ConversationalRetrievalChain: A configured conversational chain for RAG.
    """
    llm = ChatOpenAI(temperature=temperature, model_name=model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    if system_prompt:
        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message = HumanMessagePromptTemplate.from_template(
            "Context: {context}\n\nChat History: {chat_history}\n\nHuman: {question}\nAssistant:"
        )
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": chat_prompt},
        )
    else:
        return ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=retriever, memory=memory
        )
