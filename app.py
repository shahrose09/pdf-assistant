
import os
import tempfile
import hashlib

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="AI PDF Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI PDF Research Assistant")
st.write(
    "Upload your PDFs, ask questions, and get answers "
    "based on the information in your documents."
)


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "indexed_signature" not in st.session_state:
    st.session_state.indexed_signature = None

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("📄 Your Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.caption(
        "Upload text-based PDFs. Scanned PDFs may require OCR."
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
file_data = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue()

        signature = hashlib.sha256(content).hexdigest()

        file_data.append({
            "name": uploaded_file.name,
            "content": content,
            "signature": signature
        })

current_signature = tuple(
    (item["name"], item["signature"])
    for item in file_data
)



if not file_data:
    st.session_state.vectorstore = None
    st.session_state.indexed_signature = None

elif current_signature != st.session_state.indexed_signature:

   
    st.session_state.vectorstore = None

    all_documents = []
    processing_errors = []

    with st.spinner("Reading and processing your PDFs..."):

        for item in file_data:
            pdf_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:
                    temp_file.write(item["content"])
                    pdf_path = temp_file.name

                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                for document in documents:
                    document.metadata["source"] = item["name"]

                all_documents.extend(documents)

            except Exception as error:
                processing_errors.append(
                    f"{item['name']}: {error}"
                )

            finally:
                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)

        # Divide long documents into smaller overlapping chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(all_documents)

        # Remove empty chunks.
        chunks = [
            chunk for chunk in chunks
            if chunk.page_content.strip()
        ]

        if chunks:
            try:
                with st.spinner(
                    "Creating embeddings and searchable index..."
                ):
                    embeddings = HuggingFaceEmbeddings(
                        model_name=(
                            "sentence-transformers/all-MiniLM-L6-v2"
                        )
                    )

                    st.session_state.vectorstore = (
                        FAISS.from_documents(chunks, embeddings)
                    )

                st.session_state.indexed_signature = (
                    current_signature
                )

                st.success(
                    f"Ready! Processed {len(all_documents)} pages "
                    f"into {len(chunks)} text chunks."
                )

            except Exception as error:
                st.error(
                    "Could not create the search index. "
                    "Check your installed packages and connection."
                )
                st.caption(str(error))

        else:
            st.warning(
                "No readable text was found in the uploaded PDFs."
            )

            st.session_state.indexed_signature = current_signature

    for error in processing_errors:
        st.warning(f"Could not process one PDF: {error}")




api_key = os.getenv("GROQ_API_KEY")

llm = None

if api_key:
    try:
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=api_key,
            temperature=0,
            max_retries=2
        )
    except Exception as error:
        st.error("Could not initialize Groq.")
        st.caption(str(error))
else:
    st.warning(
        "Groq API key not found. "
        "Check your .env file."
    )


st.divider()
st.subheader("💬 Chat with your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



question = st.chat_input(
    "Ask a question about your uploaded PDFs..."
)

if question:

    # Display and save the user's question.
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.vectorstore is None:

        answer = (
            "Please upload readable PDFs and wait for processing "
            "to finish before asking a question."
        )

    elif llm is None:

        answer = (
            "Groq is not configured. Check that GROQ_API_KEY "
            "is set correctly in your .env file, then restart "
            "the app."
        )

    else:
        try:
            with st.spinner(
                "Searching your PDFs and generating an answer..."
            ):

                # Retrieve the three most relevant chunks.
                results = (
                    st.session_state.vectorstore.similarity_search(
                        question,
                        k=3
                    )
                )

                if not results:
                    answer = (
                        "I couldn't find relevant information "
                        "in the uploaded PDFs."
                    )

                else:
                    # Build the evidence sent to Groq.
                    context_parts = []

                    for document in results:
                        filename = document.metadata.get(
                            "source", "Unknown PDF"
                        )

                        page = document.metadata.get("page")

                        if page is not None:
                            page_number = int(page) + 1
                        else:
                            page_number = "Unknown"

                        context_parts.append(
                            f"[Source: {filename}, "
                            f"Page: {page_number}]\n"
                            f"{document.page_content}"
                        )

                    context = "\n\n".join(context_parts)

                    prompt = f"""
You are an AI PDF Research Assistant.

Answer the user's question using the supplied PDF excerpts.

Rules:
1. Use the excerpts as your evidence.
2. If the evidence is insufficient, say so clearly.
3. Never invent facts, quotations, filenames, or page numbers.
4. Cite factual claims using the provided filename and page,
   for example: (research.pdf, Page 3).
5. If the excerpts disagree, explain the disagreement.
6. Write a clear, useful answer.

PDF EXCERPTS:
{context}

USER QUESTION:
{question}

ANSWER:
"""

                    response = llm.invoke(prompt)

                    answer = response.content

                    # Handle text and structured content responses.
                    if not isinstance(answer, str):
                        answer = str(answer)

        except Exception as error:
            answer = (
                "Sorry, I couldn't generate an answer. Check your "
                "internet connection, Groq API key, model access, "
                "and API usage limits.\n\n"
                f"Technical details: {error}"
            )

    # Display and save the assistant's response.
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
#.\.venv\Scripts\python.exe -m streamlit run app.py

