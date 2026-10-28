
import os
import shutil
from pathlib import Path
from datetime import datetime

import chromadb
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import Any, cast
from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from crewai_tools import SerperDevTool


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAG_DIR = BASE_DIR / "rag_db"
ANSWERS_FILE = BASE_DIR / "answers.txt"

DATA_DIR.mkdir(exist_ok=True)
RAG_DIR.mkdir(exist_ok=True)

COLLECTION_NAME = "customer_support_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ============================================================
# 2. VALIDATE CONFIGURATION
# ============================================================

def validate_api_keys():
    missing = []

    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if not SERPER_API_KEY:
        missing.append("SERPER_API_KEY")

    if missing:
        st.error(
            "Missing environment variables: "
            + ", ".join(missing)
            + ". Add them to your .env file and restart the app."
        )
        st.stop()


# ============================================================
# 3. PERSISTENT RAG DATABASE
# ============================================================

@st.cache_resource
def get_embedding_function():
    # Downloads the model the first time if it is not cached.
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Chroma types this parameter for all embeddables, including images,
    # while this application intentionally stores text documents only.
    return cast(EmbeddingFunction[Embeddable], embedding_function)


@st.cache_resource
def get_chroma_client():
    # Data remains on disk across application restarts.
    return chromadb.PersistentClient(path=str(RAG_DIR))


def get_collection():
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"description": "Customer support knowledge base"},
    )


def index_pdf(uploaded_file):
    """Index an uploaded PDF with LangChain Community loaders and splitters."""
    import hashlib

    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    pdf_path = DATA_DIR / uploaded_file.name
    pdf_path.write_bytes(pdf_bytes)

    collection = get_collection()

    # Skip an identical PDF that has already been indexed.
    existing = collection.get(
        where={"file_hash": file_hash},
        limit=1,
    )

    if existing["ids"]:
        return (
            f"'{uploaded_file.name}' is already indexed. "
            "No duplicate chunks were added."
        )

    # DirectoryLoader makes this compatible with directory-based ingestion,
    # while the glob limits this upload to its own PDF.
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob=uploaded_file.name,
        # LangChain's DirectoryLoader type stub does not yet include
        # PyPDFLoader, although it is a supported runtime loader class.
        loader_cls=cast(Any, PyPDFLoader),
        show_progress=False,
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)
    all_chunks = []

    for chunk_number, chunk in enumerate(chunks):
        page_number = chunk.metadata.get("page", 0) + 1
        all_chunks.append(
            {
                "id": f"{file_hash}_p{page_number}_c{chunk_number}",
                "text": chunk.page_content.strip(),
                "metadata": {
                    "filename": uploaded_file.name,
                    "file_hash": file_hash,
                    "page": page_number,
                    "loader": "langchain_community.PyPDFLoader",
                },
            }
        )

    if not all_chunks:
        return (
            "No extractable text was found in this PDF. "
            "If it is scanned, run OCR first."
        )

    # Add in batches to handle larger PDFs.
    for batch_start in range(0, len(all_chunks), 100):
        batch = all_chunks[batch_start:batch_start + 100]

        collection.add(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )

    return (
        f"Indexed {len(all_chunks)} text chunks "
        f"from '{uploaded_file.name}'."
    )


def search_knowledge_base(question, top_k=4):
    """Retrieve relevant passages from the persistent vector DB."""
    collection = get_collection()

    if collection.count() == 0:
        return (
            "The customer-support knowledge base is empty. "
            "The user has not indexed a document yet."
        )

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Chroma's response type allows these fields to be None when they were
    # not returned, so normalize missing values before indexing the first
    # query's results.
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    if not documents:
        return "No relevant passages were found in the knowledge base."

    passages = []

    for document, metadata in zip(documents, metadatas):
        passages.append(
            f"Source: {metadata.get('filename', 'Unknown')}, "
            f"page {metadata.get('page', '?')}\n{document}"
        )

    return "\n\n---\n\n".join(passages)


# ============================================================
# 4. CUSTOM RAG TOOL FOR CREWAI
# ============================================================

@tool("Customer Support Knowledge Base Search")
def search_customer_support_knowledge_base(question: str) -> str:
    """Search the customer-support PDF knowledge base for company policies,
    FAQs, troubleshooting instructions, and other documented information.
    If the knowledge base is empty, report that clearly.
    """
    try:
        return search_knowledge_base(question)
    except Exception as exc:
        return f"Knowledge-base search failed: {exc}"


# ============================================================
# 5. FILE-WRITING TOOL FOR THE ENTRY AGENT
# ============================================================

@tool("Save Customer Support Answers")
def save_customer_support_answers(
    query: str,
    answer1: str,
    answer2: str,
) -> str:
    """Save the original query and both complete assistant answers to
    answers.txt.
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with ANSWERS_FILE.open("a", encoding="utf-8") as file:
            file.write("\n" + "=" * 70 + "\n")
            file.write(f"Timestamp: {timestamp}\n")
            file.write(f"Customer Query:\n{query}\n\n")
            file.write(f"Assistant Answer:\n{answer1}\n\n")
            file.write(f"Web Search Assistant Answer:\n{answer2}\n")

        return f"Both answers saved to {ANSWERS_FILE.name}."
    except Exception as exc:
        return f"Could not save answers: {exc}"


# ============================================================
# 6. BUILD THE THREE CREWAI AGENTS
# ============================================================

def build_crew(customer_query):
    rag_tool = search_customer_support_knowledge_base
    web_tool = SerperDevTool()
    save_tool = save_customer_support_answers

    assistant = Agent(
        role="Assistant",
        goal=(
            "Answer the customer's query clearly and accurately, "
            "using the customer-support knowledge base for company-specific "
            "policies and information."
        ),
        backstory=(
            "You are the first-line customer support assistant. "
            "Consult the knowledge-base search tool for relevant information. "
            "Use the retrieved passages as evidence for company policies. "
            "If the database is empty or the answer is not documented, "
            "say so clearly instead of inventing company policy. "
            "You may provide general knowledge, but distinguish it from "
            "verified company information."
        ),
        tools=[rag_tool],
        llm=OPENAI_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    web_assistant = Agent(
        role="Web Search Assistant",
        goal=(
            "Search the web for the customer's query and provide a "
            "separate, useful answer supported by web results."
        ),
        backstory=(
            "You are a web research specialist. Use the Serper web-search "
            "tool to find relevant and current information. Summarize the "
            "results accurately, distinguish sourced facts from uncertainty, "
            "and include useful source URLs when available. Do not simply "
            "repeat the first assistant's answer."
        ),
        tools=[web_tool],
        llm=OPENAI_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    entry_agent = Agent(
        role="Entry Agent",
        goal=(
            "Record the customer query and both previous agent answers "
            "in a text file, ensuring the complete answers are preserved."
        ),
        backstory=(
            "You maintain customer-support records. Read the outputs of "
            "both previous tasks, then call the file-saving tool with the "
            "original query and both complete answers. Never invent or "
            "alter the answers. Confirm whether the save succeeded."
        ),
        tools=[save_tool],
        llm=OPENAI_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    # Task 1: RAG-backed support answer.
    task1 = Task(
        description=(
            "Answer the following customer query:\n\n"
            "{customer_query}\n\n"
            "Use the Customer Support Knowledge Base Search tool to "
            "retrieve relevant passages. Answer from the evidence where "
            "possible. Clearly disclose when the knowledge base has no "
            "relevant information. Do not invent company policies."
        ),
        expected_output=(
            "A clear customer-facing answer, grounded in retrieved "
            "support-document passages where available. State any "
            "important knowledge-base limitations."
        ),
        agent=assistant,
    )

    # Task 2: Independent web research answer.
    task2 = Task(
        description=(
            "Research this original customer query using web search:\n\n"
            "{customer_query}\n\n"
            "Use SerperDevTool to search for relevant information. "
            "Produce your own answer based on the search results, "
            "including source URLs where available. This must be a "
            "separate web-researched answer, not a copy of the previous "
            "agent's answer."
        ),
        expected_output=(
            "A separate answer based on web research, with useful source "
            "URLs and uncertainty clearly identified."
        ),
        agent=web_assistant,
        context=[task1],
    )

    # Task 3: Save both preceding task outputs.
    task3 = Task(
        description=(
            "Save the following customer-support record to answers.txt.\n\n"
            "Original customer query:\n{customer_query}\n\n"
            "The context contains the complete outputs of Task 1 "
            "(Assistant) and Task 2 (Web Search Assistant). Extract both "
            "answers faithfully. Call the Save Customer Support Answers "
            "tool with the original query, the complete Task 1 answer, "
            "and the complete Task 2 answer. Do not substitute your own "
            "summary for either answer. Report whether saving succeeded."
        ),
        expected_output=(
            "Confirmation that the original query and both complete "
            "answers have been saved to answers.txt."
        ),
        agent=entry_agent,
        context=[task1, task2],
    )

    return Crew(
        agents=[assistant, web_assistant, entry_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True,
    )


# ============================================================
# 7. STREAMLIT USER INTERFACE
# ============================================================

st.set_page_config(
    page_title="CrewAI Customer Support",
    page_icon="💬",
    layout="wide",
)

st.title("💬 CrewAI Customer Support Agent")
st.caption(
    "Three sequential agents • Persistent local RAG • Web search"
)

with st.sidebar:
    st.header("Knowledge Base")

    st.write(
        "Upload a customer-support PDF to add it to the local "
        "persistent RAG database."
    )

    uploaded_pdf = st.file_uploader(
        "Choose a customer-support PDF",
        type=["pdf"],
    )

    if st.button(
        "Index PDF",
        disabled=uploaded_pdf is None,
        use_container_width=True,
    ):
        try:
            with st.spinner("Extracting, embedding, and indexing PDF..."):
                message = index_pdf(uploaded_pdf)
            st.success(message)
        except Exception as exc:
            st.error(f"PDF indexing failed: {exc}")

    try:
        count = get_collection().count()
        st.metric("Indexed text chunks", count)
    except Exception:
        st.info("The RAG database will be initialized when needed.")

    st.divider()
    st.subheader("Database management")

    confirm_delete = st.checkbox(
        "I understand this deletes the indexed knowledge base."
    )

    if st.button(
        "Delete RAG Database",
        disabled=not confirm_delete,
        use_container_width=True,
    ):
        try:
            client = get_chroma_client()

            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass

            st.cache_resource.clear()
            st.success(
                "The indexed collection has been deleted. "
                "Your original PDF files have not been deleted."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Could not delete the collection: {exc}")

    st.caption(f"Database folder: {RAG_DIR}")
    st.caption(f"Answers file: {ANSWERS_FILE}")


# Query input
customer_query = st.text_area(
    "Enter your customer-support query",
    placeholder="Example: What is your refund policy?",
    height=100,
)

if st.button("Submit Query", type="primary", use_container_width=True):
    if not customer_query.strip():
        st.warning("Please enter a question first.")
    else:
        validate_api_keys()

        with st.spinner(
            "CrewAI is working: Assistant → Web Search Assistant → Entry Agent..."
        ):
            try:
                crew = build_crew(customer_query.strip())

                result = crew.kickoff(
                    inputs={"customer_query": customer_query.strip()}
                )

                # Get each answer directly from its corresponding task.
                assistant_answer = (
                    crew.tasks[0].output.raw
                    if crew.tasks[0].output
                    else "No answer was returned by the Assistant."
                )

                web_answer = (
                    crew.tasks[1].output.raw
                    if crew.tasks[1].output
                    else "No answer was returned by the Web Search Assistant."
                )

                entry_answer = (
                    crew.tasks[2].output.raw
                    if crew.tasks[2].output
                    else "The Entry Agent did not return a result."
                )

                # Safety fallback: preserve the two answers even if the
                # Entry Agent fails to call its saving tool.
                # A successful tool call already appends a record, so do
                # not append again when the expected record is present.
                existing_text = (
                    ANSWERS_FILE.read_text(encoding="utf-8")
                    if ANSWERS_FILE.exists()
                    else ""
                )

                if (
                    customer_query.strip() not in existing_text
                    or assistant_answer not in existing_text
                    or web_answer not in existing_text
                ):
                    with ANSWERS_FILE.open("a", encoding="utf-8") as file:
                        file.write("\n" + "=" * 70 + "\n")
                        file.write(
                            "Timestamp: "
                            + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            + "\n"
                        )
                        file.write(
                            f"Customer Query:\n{customer_query.strip()}\n\n"
                        )
                        file.write(
                            f"Assistant Answer:\n{assistant_answer}\n\n"
                        )
                        file.write(
                            f"Web Search Assistant Answer:\n{web_answer}\n"
                        )

                st.session_state["assistant_answer"] = assistant_answer
                st.session_state["web_answer"] = web_answer
                st.session_state["entry_answer"] = entry_answer
                st.session_state["query"] = customer_query.strip()

            except Exception as exc:
                st.error(f"The crew could not complete the request: {exc}")


# Display answers from the most recent run.
if "assistant_answer" in st.session_state:
    st.divider()
    st.subheader("Results")

    st.markdown("### 1. Assistant — Customer Support / RAG")
    st.markdown(st.session_state["assistant_answer"])

    st.markdown("### 2. Web Search Assistant")
    st.markdown(st.session_state["web_answer"])

    st.markdown("### 3. Entry Agent")
    st.write(st.session_state["entry_answer"])

    if ANSWERS_FILE.exists():
        st.success(f"Record saved to {ANSWERS_FILE.name}")

        with st.expander("View saved records"):
            st.text(ANSWERS_FILE.read_text(encoding="utf-8"))

    st.download_button(
        "Download answers.txt",
        data=ANSWERS_FILE.read_text(encoding="utf-8")
        if ANSWERS_FILE.exists()
        else "",
        file_name="answers.txt",
        mime="text/plain",
    )
