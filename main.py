from langchain.memory import ConversationBufferMemory
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.memory import ChatMessageHistory

def initializemodel():
    llm = CTransformers(model="../model/llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama",
                        config={'max_new_tokens': 256,'temperature': 0.01})

    return llm


def createtemplatemessage(model):

    demo_ephemeral_chat_history = ChatMessageHistory()

    # Create template message
    template_messages = [
        SystemMessage(content="You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(template_messages)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)

    response=chain.run(text="What can I see in Vienna? Propose a few locations. Names only, no details.")
    print(response)

    return response


if __name__ == "__main__":
    initializemodel()
    createtemplatemessage(initializemodel())
