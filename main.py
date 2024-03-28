import os
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

user_chat_history = ChatMessageHistory()

load_dotenv()  # Lädt die Variablen aus der .env-Datei
api_key = os.getenv("OPENAI_KEY")


def initializemodel():
    chat = CTransformers(model="../model/llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama",
                         config={'max_new_tokens': 256, 'temperature': 0.01})

    return chat


def chromadb():
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    data = loader.load()

    # Angenommen, 'data' ist die Liste, die du erhalten hast
    document_content = data[
        0].page_content  # Zugriff auf das erste (und einzige) Element in der Liste und sein page_content Attribut

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=100,
        chunk_overlap=0,
    )

    doc = [
        Document(page_content=f"{document_content}")
    ]

    splits = text_splitter.split_documents(doc)

    print(splits)
    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)

    # query it
    query = "Can LangSmith help test my LLM applications?"
    docs = vectorstore.similarity_search(query)

    # print results
    print(docs[0].page_content)

    retriever = vectorstore.as_retriever(k=4)

    injection = retriever.invoke("Can LangSmith help test my LLM applications?")

    print(injection)

    # Extrahiere 'page_content' aus jedem Document in 'injection' und füge sie zu einem einzigen String zusammen
    injection = ''.join(doc.page_content for doc in injection)

    # Ausgabe des kombinierten Inhalts
    print(injection)

    return injection


def createtemplatemessage(chat, injection):

    print(injection)
    # Create template message
    template_messages = [
        SystemMessage(content="You translate everything to german. And take this in consideration"+injection),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]

    prompt = ChatPromptTemplate.from_messages(template_messages)
    print(prompt)

    chain = prompt | chat

    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: user_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return chain_with_message_history


def invokechain(chain, userinput):

    response = chain.invoke(
        {"input": userinput},
        {"configurable": {"session_id": "unused"}},
    )

    return response


if __name__ == "__main__":
    chain = createtemplatemessage(initializemodel(), chromadb())
    response = invokechain(chain, "Can LangSmith help test my LLM applications?")
    print(response)

