**📄 RAG Q&A Conversation with PDFs (Chat History Enabled)**

A Retrieval-Augmented Generation (RAG) based conversational application that allows users to upload PDF documents and chat with them, while maintaining conversation history across sessions.

Built using LangChain’s latest runnable architecture, Groq LLM (LLaMA), Hugging Face embeddings, Chroma vector store, and Streamlit for the UI.

________________________________________________________________________

**🚀 Features**

📂 Upload multiple PDF files

🔍 Semantic search using vector embeddings

🧠 History-aware RAG (understands follow-up questions)

💬 Persistent chat history per session

⚡ Fast inference using Groq LLaMA models

🖥️ Simple and interactive Streamlit UI

❌ Hallucination-aware responses (answers only from retrieved context)

________________________________________________________________________

**🛠️ Tech Stack**

Python

LangChain (LCEL & Runnables)

Groq LLM (LLaMA 3.3 – 70B)

Hugging Face Embeddings (all-MiniLM-L6-v2)

Chroma Vector Store

Streamlit

PyPDFLoader

________________________________________________________________________

**🧠 Architecture Overview**

User Query
   ↓
Chat History Aware Retriever
   ↓
Vector Store (Chroma)
   ↓
Relevant Context
   ↓
LLM (Groq - LLaMA)
   ↓
Concise Answer
   ↓
Stored in Session Chat History

________________________________________________________________________

**📂 Project Structure**

├── app.py                  # Streamlit application
├── requirements.txt        # Project dependencies
├── .env.example            # Environment variable template
├── README.md               # Documentation
________________________________________________________________________

**🧪 How It Works**

Upload one or more PDF files

PDFs are split into chunks

Embeddings are generated using Hugging Face

Stored in Chroma vector database

User asks a question

Retriever fetches relevant context

LLM answers only from retrieved content

Chat history is maintained across queries
________________________________________________________________________

**🧠 Key Concepts Implemented**

Retrieval-Augmented Generation (RAG)

History-aware retrieval

LangChain RunnableWithMessageHistory

Vector similarity search

Controlled context to reduce hallucinations

Session-based memory management
________________________________________________________________________

**📘 Learning Outcomes**

Implemented conversation-aware RAG pipelines

Understood LangChain agent & retriever internals

Learned how to manage chat history with LCEL

Built safe, production-style GenAI applications

Integrated Groq LLMs with LangChain
