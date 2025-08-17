import os
import dotenv
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

MODEL = 'llama-3.3-70b-versatile'
llm = ChatGroq(
    temperature=0,
    model_name=MODEL
)

# Path to save/load FAISS index
FAISS_INDEX_PATH = "faiss_index"

# Initialize embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name = model_name)

# Initialize Streamlit app
st.header("Retrieval-Augmented Generation (RAG) based AI-Powered PDF Question-Answering Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Check if FAISS index exists
vector_store = None
if os.path.exists(FAISS_INDEX_PATH):
    # Load the existing FAISS index
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    st.write("Loaded existing FAISS index.")

# Process uploaded PDF
if file:
    text = extract_text_from_pdf(file)
    splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    st.write(chunks)
    st.write(f"Total chunks created: {len(chunks)}")

    # Create new FAISS index if not already loaded
    if vector_store is None:
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        st.write("Created and saved new FAISS index with uploaded PDF.")

# Allow question input if vector store is available
if vector_store is not None:
    # Initialize chat history in session_state if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("What do you want to ask?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            model=MODEL,
        )       
    
    # Perform similarity search when user asks a question
    if prompt:
        question_embedding = embeddings.embed_query(prompt)
        match = vector_store.similarity_search_by_vector(question_embedding)
        qa_chain = load_qa_chain(llm, chain_type="stuff")
        answer = qa_chain.run(input_documents=match, question=prompt)

        # Display and store LLM response
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.write("Please upload a PDF to create or load the FAISS index.")
