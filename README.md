# CrewAI Customer Support App

A Streamlit customer-support application using three CrewAI agents, a local Chroma knowledge base, LangChain document loaders, and Serper web search.

## Features

- Upload and index customer-support PDFs.
- Load PDFs with LangChain Community `DirectoryLoader` and `PyPDFLoader`.
- Split documents with `RecursiveCharacterTextSplitter`.
- Store embeddings persistently in ChromaDB.
- Use a CrewAI RAG tool for company-specific answers.
- Use CrewAI `SerperDevTool` for independent web research.
- Save customer queries and answers to `answers.txt`.

## Requirements

- Python 3.10+
- OpenAI API key
- Serper API key

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file in the project directory:

```env
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
OPENAI_MODEL=gpt-4o-mini
```

Never commit `.env` or expose API keys publicly.

## Run the app

This is a Streamlit application. Start it with:

```bash
streamlit run app.py
```

Open the URL displayed in the terminal, usually:

```text
http://localhost:8501
```

Do not launch Uvicorn manually; Streamlit starts its internal server.

## Usage

1. Upload a customer-support PDF in the sidebar.
2. Click **Index PDF**.
3. Enter a customer question.
4. Click **Submit Query**.
5. Review the RAG answer, web-researched answer, and saved-answer confirmation.

The persistent Chroma database is stored in `rag_db/`. Uploaded PDFs are stored in `data/`, and answer records are written to `answers.txt`.

## Troubleshooting

### Serper returns `403 Forbidden`

The web-search agent uses CrewAI's `SerperDevTool`, which calls Serper's search API. Check that `SERPER_API_KEY` is valid, active, and has available quota, then restart Streamlit.

### The page appears blank

Confirm that the terminal shows the Streamlit server is running, then open `http://localhost:8501` or `http://127.0.0.1:8501` and refresh the page.

### First launch is slow

The first launch may download and initialize the Sentence Transformer embedding model. Later launches should be faster.
