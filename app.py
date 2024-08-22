import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader, TextLoader
import pdfplumber
import requests
from io import BytesIO
import time
import os

# Streamlit APP Configuration
st.title("Summarizer")
st.subheader('Summarize Content from Multiple Sources')

# Get the Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Input fields for URLs and topic/title
topic_title = st.text_input("Topic or Title")


# Function to process URLs
def process_urls(urls):
    docs = []
    if urls:
        for url in urls:
            if "youtube.com" in url:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                video_id = url.split("v=")[-1]
                docs.extend(loader.load())
            elif validators.url(url):
                loader = UnstructuredURLLoader(
                    urls=[url],
                    ssl_verify=False,
                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                )
                docs.extend(loader.load())
                # st.markdown(f"*Website URL:* [Link]({url})")
            else:
                st.error(f"Invalid URL: {url}")
    return docs

# Function to process PDFs
def process_pdfs(pdf_files):
    docs = []
    if pdf_files:
        for pdf_file in pdf_files:
            with pdfplumber.open(BytesIO(pdf_file.read())) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
                docs.append(Document(page_content=text))
    return docs

# Function to process Text files
def process_texts(text_files):
    docs = []
    if text_files:
        for text_file in text_files:
            text = text_file.read().decode("utf-8")
            docs.append(Document(page_content=text))
    return docs

# Function to summarize documents
def summarize_docs(docs, topic_title):
    summaries = []
    try:
        llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

        # Define the prompt template
        prompt_template = f"""
        Provide a summary of the following content focusing on the topic: "{topic_title}":
        Content:{{text}}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

        # Chain for Summarization
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

        # Process content in batches
        max_tokens_per_batch = 500  # Adjust this as needed based on rate limits
        batch_size = 5  # Number of documents per batch
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            try:
                output_summary = chain.run(batch)
                summaries.append(output_summary)
            except Exception as e:
                st.exception(f"An error occurred during summarization: {e}")
                if "rate limit" in str(e).lower():
                    st.info("Rate limit reached. Retrying after a pause...")
                    time.sleep(300)  # Wait for 5 minutes before retrying

        combined_summary = "\n".join(summaries)
        return combined_summary

    except Exception as e:
        st.exception(f"An error occurred: {e}")
        return None



# Selectbox for choosing the summarization type
summarization_type = st.selectbox(
    "Choose what to summarize:",
    ("URLs", "PDF Files", "Text Files")
)

# Text area for URLs input in the sidebar if URLs are selected
if summarization_type == "URLs":
    input_urls = st.text_area("Enter URLs (one per line)")

# File uploaders if PDFs or Text Files are selected
elif summarization_type == "PDF Files":
    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
elif summarization_type == "Text Files":
    text_files = st.file_uploader("Upload Text files", type=["txt"], accept_multiple_files=True)

# Main content area for action buttons
st.title("Summarization Actions")

# Button to summarize based on the selected type
if st.button(f"Summarize {summarization_type}"):
    if not groq_api_key.strip() or not topic_title.strip():
        st.error("Please provide the Groq API Key and a topic/title to get started.")
    else:
        if summarization_type == "URLs":
            urls = [url.strip() for url in input_urls.split('\n') if url.strip()]
            docs = process_urls(urls)
        elif summarization_type == "PDF Files":
            docs = process_pdfs(pdf_files)
        elif summarization_type == "Text Files":
            docs = process_texts(text_files)
        
        # If documents are found, proceed to summarize
        if docs:
            summary = summarize_docs(docs, topic_title)
            if summary:
                st.success(f"Summary generated successfully for {summarization_type}!")
                st.write(summary)               

# Additional styling for a better user experience
st.markdown("""
    <style>
    .css-1g6z4e0 {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)