import os
import logging
import datetime
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY1 = os.getenv("OPENAI_API_KEY1")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY1

# Set directories
PDF_DIRECTORY = "./pdf_storage"
PERSIST_DIRECTORY = "./storage"
LOGS_DIRECTORY = "./logs"

# Ensure directories exist
os.makedirs(PDF_DIRECTORY, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(LOGS_DIRECTORY, exist_ok=True)  # Create logs directory

# Generate log filename based on current date (e.g., logs/app_2025-01-30.log)
log_date = datetime.datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOGS_DIRECTORY, f"app_{log_date}.log")

# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

# Streamlit UI
st.title("PDF ChatBot")
logging.info("Application Started")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join(PDF_DIRECTORY, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Uploaded and saved file: {uploaded_file.name}")
    logging.info(f"File uploaded: {uploaded_file.name} on {log_date}")

    # Process PDF
    st.write(f"Processing PDF: {file_path}")
    logging.info(f"Processing started for: {file_path}")

    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
        texts = text_splitter.split_documents(documents)
        logging.info(f"Successfully split {len(texts)} text chunks from {uploaded_file.name}")

        # Create embeddings and store them in ChromaDB
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
        vectordb.persist()
        logging.info(f"Embeddings stored in ChromaDB for {uploaded_file.name}")

        # Define retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Define LLM model
        llm = ChatOpenAI(model_name='gpt-4')

        # Create RetrievalQA chain
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        logging.info("RetrievalQA chain created successfully")

        # Let the user ask a question
        user_question = st.text_input("Ask a question about the PDF:")

        if user_question:
            logging.info(f"User query at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {user_question}")

            with st.spinner("Generating response..."):
                try:
                    llm_response = qa({"query": user_question})
                    response_text = llm_response.get('result', 'No response generated.')

                    # Display the result
                    st.write("### ChatGPT Response:")
                    st.write(response_text)
                    logging.info(f"Generated response for query: {user_question}")

                except Exception as err:
                    error_message = f"Error while generating response: {str(err)}"
                    st.error(error_message)
                    logging.error(error_message)

    except Exception as e:
        error_message = f"Error processing PDF: {str(e)}"
        st.error(error_message)
        logging.error(error_message)
