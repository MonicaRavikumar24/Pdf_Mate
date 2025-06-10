import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain_huggingface import HuggingFaceEndpoint
from templates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.7,
        max_new_tokens=500,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)


def handle_userinput(user_question):
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response.get('chat_history', [])

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            


    except StopIteration:
        st.error("The model did not return a valid response. This might be due to an issue with the Hugging Face endpoint or the question being too vague.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


def main():
    load_dotenv()

    st.set_page_config(
        page_title="Pdf Mate",
        page_icon="https://i.ibb.co/Gfkn1bMk/cartoon-style-robot-vectorart-78370-4103.jpg"
    )
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "chat_ready" not in st.session_state:
        st.session_state.chat_ready = False
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Always show the title
    st.markdown(
    """
    <div style='text-align: center;margin-top: 1px; margin-bottom: 5px;'>
        <h5 style='display: inline-flex; align-items: center; gap: 10px; justify-content: center; font-size: 2.2em;'>
            Pdf Mate
            <img src='https://i.ibb.co/Gfkn1bMk/cartoon-style-robot-vectorart-78370-4103.jpg' width='45' style='border-radius: 50%; object-fit: cover;' />
        </h5>
    </div>
    """,
    unsafe_allow_html=True
)

    # If not ready, show upload interface
    if not st.session_state.chat_ready:
        st.subheader("Drop your PDFs here and start a smart conversation with your documents.")
    

        
        
        pdf_docs = st.file_uploader("Drag and drop or browse PDF files", accept_multiple_files=True)

        if st.button("Start Chatting"):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    st.session_state.uploaded_files_info = [
                        {"name": f.name, "size": f"{round(len(f.read()) / 1024, 2)} KB"} for f in pdf_docs
                    ]
                    for f in pdf_docs:
                        f.seek(0)
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.chat_ready = True
                    st.rerun()
            else:
                st.warning("⚠️ Please upload at least one PDF before continuing.")
        return  

    # ======= MAIN CHAT INTERFACE =======

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation:
            handle_userinput(user_question)
        else:
            st.warning("⚠️ Please upload and process your PDFs first.")

    with st.sidebar:
        st.subheader("Uploaded documents")
        if st.session_state.get("uploaded_files_info"):
            for file in st.session_state.uploaded_files_info:
                st.markdown(f"🔹 **{file['name']}** — `{file['size']}`")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("✅ Processing complete! You can now ask questions.")
            else:
                st.warning("⚠️ Please upload at least one PDF before processing.")



if __name__ == "__main__":
    main()  