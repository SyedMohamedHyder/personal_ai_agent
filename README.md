# personal_ai_agent

An AI-powered agent that uses Retrieval-Augmented Generation (RAG) with OpenAI GPT to answer questions based on personal profile data. It intelligently retrieves relevant information from a custom knowledge base and responds through a conversational chat interface.

---

## 🚀 What This Repo Does

- Loads structured markdown documents from a personal knowledge base (e.g., LinkedIn data).
- Splits and embeds documents using OpenAI embeddings.
- Stores vectors in a persistent Chroma vector store.
- Retrieves relevant chunks using a vector search.
- Passes results into a GPT-powered conversational agent.
- Provides a clean chat interface (Gradio) with optional visualizations.

---

## 📂 How to Place Your Data

1. Put your profile-related `.md` files into subfolders inside:

    ```
    data/knowledge_base/
    ```

2. Organize by category (each folder becomes a `doc_type`):

    ```
    data/knowledge_base/
    ├── profile/
    │   └── about_me.md
    ├── skills/
    │   └── technical_skills.md
    ├── projects/
    │   └── project_alpha.md
    ```

3. Each `.md` file will be parsed, tagged by folder name, and indexed for retrieval.

---

## ▶️ How to Run the App

### 1. Install dependencies

```bash
uv pip install -r requirements.txt
```

> 💡 This project uses [uv](https://github.com/astral-sh/uv) — a faster Python package manager. You can also use `pip` if preferred.

### 2. Set your OpenAI API key

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your-openai-key-here
```

### 3. Launch the app

```bash
uv run app/app.py
```

This will:

- Index your documents
- Start the RAG pipeline
- Open the chat interface in your default browser (dark theme)

---

## 🛠 Developer Notes

- Vector store: [Chroma](https://www.trychroma.com/)
- Embeddings & LLM: [OpenAI GPT](https://platform.openai.com/docs)
- Frameworks: LangChain, Gradio
- Dependency manager: [`uv`](https://github.com/astral-sh/uv)
- Code modules:
  - `loaders/` – document loading & chunking
  - `vectorstore/` – vector DB management
  - `rag/` – LLM + retriever chain
  - `ui/` – visualization tools
  - `notebooks/` – Jupyter notebooks for dev
- Configurable settings are stored in `config.py`

---

## 💬 Example Questions You Can Ask

- "What certifications do I have?"
- "Summarize my career background."
- "List all the technologies I've worked with."
- "What are my top skills?"

---

## 📄 License

MIT
