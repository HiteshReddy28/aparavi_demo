# Aparavi Data Toolchain Q&A Demo — Project Overview

This document provides an end‑to‑end description of the **Document Q&A Demo** project.  The demo shows how to build a retrieval‑augmented question answering (RAG) system over unstructured documents using the Aparavi Data Toolchain (DTC) and, when API access is unavailable, how to simulate the same pipeline with open‑source tools.

## 1 Use Case

Organisations often store large collections of policies, guidelines, handbooks and meeting notes.  Employees need to quickly find answers such as “What does the parental leave policy cover?” or “How do I request reimbursement?”.  Manually searching through documents is time‑consuming and error‑prone.  The goal of this project is to automate document ingestion, generate vector embeddings, store them in a vector database and query them via a large language model to answer user questions.

## 2 Demo Variants

### 2.1 Full DTC Pipeline (with API access)

If you have access to the Aparavi Data Toolchain API (available on paid plans), the demo includes:

- **`app.py`** — A Python script using the `aparavi-dtc-sdk` to define and execute a DTC ingestion pipeline (`pipeline_config.json`) and a query pipeline.  The ingestion pipeline scans documents from `sample_docs/`, indexes and classifies them, chunks and embeds the text and stores embeddings in a vector store (Qdrant).  The query pipeline embeds a question, retrieves the most relevant chunks and forwards them to an LLM (e.g. OpenAI’s ChatGPT) to generate a final answer.
- **`pipeline_config.json`** — A template for the ingestion pipeline.  You can edit this JSON or replace it with one exported from the DTC UI.
- **`sample_docs/`** — Example documents (policies and guidelines) that are ingested by default.  You may add your own files here.

**Running the full pipeline** requires API keys for Aparavi, Qdrant and your LLM provider.  After installing dependencies (`pip install aparavi-dtc-sdk qdrant-client openai`) and setting environment variables (`DTC_API_KEY`, `DTC_BASE_URL`, `QDRANT_HOST`, `OPENAI_API_KEY`), run `python app.py --ingest` to ingest documents, followed by `python app.py --ask "Your question"` to query the data.

### 2.2 Offline Simulation (no API keys)

Many users on the community tier cannot generate an API key.  To ensure the demo remains functional, this repository also includes:

- **`dtc_local_demo.py`** — A local pipeline simulator implemented with scikit‑learn and NumPy.  It reads `.txt` files from `sample_docs/`, splits each into smaller chunks, vectorises them with a TF‑IDF vectoriser, builds a nearest‑neighbour index and saves everything to `ingested_index.pkl`.  The script can then embed a natural language question and retrieve the most similar chunks, printing them as context.

**Running the offline demo** does not require any external services:

1. Navigate to the `dtc_demo/` directory.
2. Run `python dtc_local_demo.py --ingest` to build the index.  The script reports how many documents and chunks were processed.
3. Run `python dtc_local_demo.py --ask "What does the parental leave policy cover?"` to retrieve relevant passages from the indexed documents.  Alternatively, run `python dtc_local_demo.py --test` to ingest and then run a sample query in one step.

This simulation reproduces the ingestion, embedding and retrieval stages of the DTC pipeline without requiring API access or an LLM.

## 3 Project Structure

| Path / File | Purpose |
|-------------|---------|
| `app.py` | Entrypoint for running ingestion and query pipelines via the DTC API. |
| `dtc_local_demo.py` | Offline simulator using scikit‑learn; no API keys needed. |
| `pipeline_config.json` | JSON template describing the DTC ingestion pipeline. |
| `sample_docs/` | Example text documents to ingest. |
| `README.md` | Detailed instructions and design decisions. |
| `dtc_demo_project_doc.md` | This overview document. |

## 4 What You Need to Provide

- **For API‑based demos:** A valid Aparavi Data Toolchain API key and base URL (Community or Enterprise tier), a running vector store (e.g. Qdrant) with host and API key (or run Qdrant locally on port 6333), and credentials for your preferred LLM provider (OpenAI, Anthropic, etc.).  Set these as environment variables before running `app.py`.
- **For the local simulation:** Nothing beyond Python and the provided source files.  Optionally, add more `.txt` documents to `sample_docs/` to test with a larger corpus.

## 5 Steps to Run the Demo End‑to‑End

1. Clone or download the repository.  Ensure that the `dtc_demo/` folder contains `app.py`, `dtc_local_demo.py`, `pipeline_config.json` and the `sample_docs/` directory.
2. Decide whether you will use the DTC API or the offline simulator:
   - **API Demo:** Install the dependencies and export your environment variables.  Run ingestion with `python app.py --ingest`, then ask questions with `python app.py --ask "<question>"`.
   - **Offline Demo:** Simply run `python dtc_local_demo.py --ingest` followed by `python dtc_local_demo.py --ask "<question>"`.
3. Capture your screen or terminal output to create the required 5–10 minute demo video showing ingestion, retrieval and reasoning about the results.
4. Push your modified project to a public GitHub repository for submission, or send the zipped folder to reviewers.

## 6 Next Steps and Extensions

- **Enhance chunking and embeddings:** Replace the TF‑IDF vectoriser in the offline demo with a transformer‑based embedding model such as `all-MiniLM-L6-v2` from the `sentence-transformers` library for better semantic retrieval.
- **Integrate an LLM:** Connect the offline retrieval results to a local or external language model to generate full answers, turning the simulation into a complete RAG system.
- **Add classification and redaction:** Incorporate a PII detector (e.g. spaCy’s `en_core_web_lg` model or the `presidio` library) to classify and mask sensitive information before indexing.
- **Web or chat interface:** Build a simple Flask or Streamlit app on top of either pipeline to provide an interactive UI for end users.

This project demonstrates the core concepts behind the Aparavi Data Toolchain using both the official API (when available) and a fully offline fallback.  It is intended as a starting point for experimentation, learning and extension.