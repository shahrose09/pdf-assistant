# 📚 AI PDF Research Assistant

Upload one or more PDFs, ask questions, and get answers based only on your documents, with the source file and page number cited.

**🔗 Live demo:** https://pdf-assistant-9lfsb3t7rgxhjzud57sxzm.streamlit.app

## Features
- Upload multiple PDF files at once
- Ask questions in a chat interface
- Answers cite the filename and page number
- Says so when the documents don't contain the answer
- Rebuilds the search index automatically when files change

## How it works (RAG pipeline)
1. PDFs are read and split into overlapping text chunks
2. Each chunk is converted to an embedding (HuggingFace `all-MiniLM-L6-v2`)
3. Embeddings are stored in a FAISS vector index
4. Your question retrieves the 3 most relevant chunks
5. Those chunks go to a Groq LLM, which writes an answer with citations

## Tech stack
Python, Streamlit, LangChain, FAISS, HuggingFace Sentence Transformers, Groq

## Run it locally
1. Clone the repo and install packages:
   ```
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project folder:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Start the app:
   ```
   streamlit run app.py
   ```

## Notes
- Works best with text-based PDFs. Scanned PDFs need OCR first.
- Don't upload sensitive documents to the public demo.

## Author
Your Name, [LinkedIn](https://www.linkedin.com/in/your-profile)