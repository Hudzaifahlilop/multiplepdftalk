import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter 
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.vectorstores import faiss
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import time


#fungsi untuk merubah file pdf menjadi text
def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_chunk_text(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

def get_vectorstore(text_chunk):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore = faiss.FAISS.from_texts(texts=text_chunk, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=st.session_state.memory,
    )
    return conversation_chain

# def handle_prompt(prompt):
#     with st.spinner("In progress.."):
#      response = st.session_state.conversation({'question': prompt})
#      st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             chat = st.chat_message("user")
#             chat.write(message.content)
#         else:
#             chat = st.chat_message("assistant")
#             chat.write(message.content)


def main():
    load_dotenv()

    st.set_page_config(page_title='Multiple Chat Pdf', page_icon=':books:')
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.subheader('Chat', divider='rainbow')
    

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask your question")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistance_response = st.session_state.conversation.run(prompt)
            # st.markdown(assistance_response)

            for chunk in assistance_response.split():
                full_response += chunk + " "
                time.sleep(0.05)

                message_placeholder.markdown(full_response + " ")
            message_placeholder.info(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


    with st.sidebar:
        st.subheader('Your Document')
        pdf_docs = st.file_uploader('Upload Your Documents Here and Click on "process"', accept_multiple_files=True)
        if st.button("process"):
          if pdf_docs is None:
            st.warning("Please upload a document before processing.")
          else:
              with st.spinner('processing'):

                # pdf to text
                raw_text = get_pdf_text(pdf_docs)

                # text to chunk
                text_chunk = get_chunk_text(raw_text)

                #CHUNK TO EMBBEDINGS AND VECCTORSTORE
                vectorstore = get_vectorstore(text_chunk)

                #conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("file is uploaded")
              
if __name__ == '__main__':
    main()