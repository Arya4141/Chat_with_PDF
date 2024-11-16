import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
import os
from chromadb.config import Settings
from chromadb import Client

# Use a publicly available model
checkpoint = "t5-small"  # Change to a valid model identifier
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, 
    device_map='cpu', 
    torch_dtype=torch.float32
)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=False,  # Ensure deterministic outputs
        temperature=0.0
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def process_pdf(uploaded_file):
    # Save the uploaded file locally
    with open("temp_uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Use the local file path with PyPDFLoader
    loader = PyPDFLoader("temp_uploaded_file.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Create an in-memory Chroma instance
    settings = Settings(
        chroma_db_impl="duckdb+parquet",  # Default in-memory configuration
        anonymized_telemetry=False
    )
    client = Client(Settings(
        chroma_db_impl="duckdb+parquet",  # Use DuckDB with Parquet files
        persist_directory="./db"         # Directory for persistent storage
    ))
    print("Chroma initialized successfully!")
    
    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        texts,
        embedding_function=embeddings,
        client=client  # Pass the initialized client
    )
    retriever = db.as_retriever()

    # Clean up the temporary file
    os.remove("temp_uploaded_file.pdf")
    
    return retriever


def process_answer(retriever, instruction):
    llm = llm_pipeline()
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
    generated_text = qa(instruction)
    return generated_text['result']

def main():
    st.title("Dynamic PDF Q&A 🐦📄")
    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI-powered Question and Answering app that responds to questions about any uploaded PDF file.
            """
        )
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file:
        retriever = process_pdf(uploaded_file)
        st.success("PDF file processed successfully!")
        question = st.text_area("Enter your Question")
        if st.button("Ask"):
            st.info("Your Question: " + question)
            answer = process_answer(retriever, question)
            st.info("Your Answer")
            st.write(answer)

if __name__ == '__main__':
    main()
