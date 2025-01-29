import os
import glob
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY1=os.getenv("OPENAI_API_KEY1")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY1

# Set directories
PDF_DIRECTORY = "./pdf_storage"
PERSIST_DIRECTORY = "./storage"

# Ensure directories exist
os.makedirs(PDF_DIRECTORY, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

st.title("PDF ChatBot")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file
    file_path = os.path.join(PDF_DIRECTORY, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.success(f"Uploaded and saved file: {uploaded_file.name}")
    
    # Process PDF
    st.write(f"Processing PDF: {file_path}")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and store them in Chroma vector database
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
    vectordb.persist()

    # Define retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Define LLM model
    llm = ChatOpenAI(model_name='gpt-4')

    # Create RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Let the user ask a question
    user_question = st.text_input("Ask a question about the PDF:")

    if user_question:
        with st.spinner("Generating response..."):
            try:
                llm_response = qa({"query": user_question})
                response_text = llm_response.get('result', 'No response generated.')

                # Display the result
                st.write("### ChatGPT Response:")
                st.write(response_text)

                # Display source documents
                # st.write("### Source Documents:")
                # docs = llm_response.get("source_documents", [])
                # print("docs : ", docs)
                # for doc in docs:
                #     source_attribute = doc.metadata.get('source', 'Unknown Source')
                #     filename = os.path.basename(source_attribute)
                #     st.write(f"- **{filename}**")

            except Exception as err:
                st.error(f"Error occurred: {str(err)}")