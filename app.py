import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os

# Use a better model for improved responses
checkpoint = "google/flan-t5-base"  # Changed from t5-small for better quality
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
        max_length=512,  # Increased for more detailed responses
        do_sample=True,
        temperature=0.7,  # Adjusted for better creativity
        top_p=0.95,
        repetition_penalty=1.2  # Added to prevent repetition
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Increased chunk size for better context
        chunk_overlap=150,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings, persist_directory="db")
    db.persist()
    
    os.unlink(tmp_path)
    return db

def qa_llm():
    llm = llm_pipeline()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    retriever = db.as_retriever(
        search_kwargs={"k": 3}  # Limit to top 3 most relevant chunks
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def clean_answer(text):
    # Remove repetitive phrases and clean up the response
    if "Helpful Answer:" in text:
        text = text.split("Helpful Answer:")[-1].strip()
    return text.strip()

def process_answer(instruction):
    qa = qa_llm()
    # Enhance the prompt to get better responses
    enhanced_prompt = f"Based on the context, provide a clear and concise answer to: {instruction}"
    generated_text = qa(enhanced_prompt)
    answer = clean_answer(generated_text['result'])
    return answer, generated_text

def main():
    st.title("📚 PDF Question Answering System")
    
    with st.expander("ℹ️ About the App"):
        st.markdown("""
            This is a Generative AI powered Question and Answering app that responds to questions about your PDF File.
            
            **How to use:**
            1. Upload your PDF file
            2. Wait for the processing to complete
            3. Ask questions about the content of your PDF
        """)
    
    pdf_file = st.file_uploader("Upload your PDF", type=['pdf'])
    
    if pdf_file is not None:
        with st.spinner("Processing PDF... This may take a moment."):
            process_container = st.empty()
            process_container.info("Creating embeddings for your PDF...")
            db = process_pdf(pdf_file)
            process_container.empty()
            st.success("PDF processed successfully! You can now ask questions.")
        
        question = st.text_area("What would you like to know about your PDF?")
        
        if st.button("Ask Question"):
            if question:
                with st.spinner("Generating answer..."):
                    answer, metadata = process_answer(question)
                    
                    # Display the answer in a nice format
                    st.markdown("### Answer:")
                    st.markdown(f">{answer}")  # Better formatting
                    
                    # Display source information in a cleaner way
                    with st.expander("📄 Source Details"):
                        for i, doc in enumerate(metadata["source_documents"], 1):
                            st.markdown(f"""
                                **Source {i}:**
                                ```
                                {doc.page_content[:200]}...
                                ```
                                **Page:** {doc.metadata.get('page', 'N/A')}
                                ---
                            """)
            else:
                st.warning("Please enter a question.")
    else:
        st.info("👆 Please upload a PDF file to get started!")

if __name__ == '__main__':
    main()
