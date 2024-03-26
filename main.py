from langchain.memory import ConversationBufferMemory
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def initializemodel():
    chat = CTransformers(model="../model/llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama",
                         config={'max_new_tokens': 256,'temperature': 0.01})

    return chat


def cromadb():
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    print(all_splits)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key="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"))

    # k is the number of chunks to retrieve
    retriever = vectorstore.as_retriever(k=4)

    docs = retriever.invoke("how can langsmith help with testing?")




def createtemplatemessage(chat):

    # Create template message
    template_messages = [
        SystemMessage(content="You translate everything to german"),
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
    cromadb()
    #initializemodel()
    #chain = createtemplatemessage(initializemodel())
    #response=invokechain(chain, "wie heisst die haupstadt von deutschland?")
    #print(response)

