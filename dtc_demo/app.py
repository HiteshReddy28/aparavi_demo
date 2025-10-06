"""
app.py – Example script demonstrating how to use the Aparavi Data Toolchain (DTC) to ingest
documents into a vector store and query them with a large language model.

This script uses the ``aparavi_dtc_sdk`` Python package to programmatically execute DTC pipelines.
It defines two main commands:

* ``--ingest`` – Builds and executes a pipeline that scans, indexes, chunks, embeds and stores
  documents from a local folder (``sample_docs/``).
* ``--ask "question"`` – Executes a small query pipeline that embeds the user’s question,
  retrieves the most relevant chunks from the vector store and sends them to an LLM.

Before running, make sure you have the following environment variables set:

* ``DTC_API_KEY`` – Your API key for the Aparavi Data Toolchain.
* ``DTC_BASE_URL`` – (Optional) The base URL of your DTC API endpoint. Defaults to ``https://dtc.aparavi.com/api/v1``.
* ``QDRANT_HOST`` and ``QDRANT_API_KEY`` – Connection details for your Qdrant instance.
* ``OPENAI_API_KEY`` – API key for the chosen LLM provider (e.g. OpenAI).

Note: This is a skeleton implementation; you will need to adjust the pipeline definitions
in ``pipeline_config.json`` and handle exceptions, authentication and error checking as
appropriate for your environment.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Load environment variables from a .env file if present. This is optional
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # If python-dotenv is not installed, environment variables must be set manually.
    pass

# Attempt to import the Aparavi DTC SDK. If missing, instruct the user to install it.
try:
    from aparavi_dtc_sdk.client import AparaviClient  # type: ignore
except ImportError as e:  # pragma: no cover
    print("ImportError:", e)
    print(
        "aparavi_dtc_sdk or one of its dependencies is not installed."
        " Install it with 'pip install aparavi-dtc-sdk requests pandas' and retry."
    )
    AparaviClient = None  # type: ignore


def load_pipeline_config() -> dict:
    """Load the ingestion pipeline configuration from the JSON file."""
    config_path = Path(__file__).parent / "pipeline_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ingest_documents() -> None:
    """
    Run the ingestion pipeline to process documents and store embeddings.

    This version uses the newer `execute_pipeline_workflow` method of the
    Aparavi SDK, which handles validation, task creation, file upload, and
    polling automatically.  Provide your pipeline configuration via
    ``pipeline_config.json`` and store your documents in the ``sample_docs/``
    directory.  Ensure you have set ``DTC_API_KEY`` (and optionally
    ``DTC_BASE_URL``) in your environment before running.  The default
    ``DTC_BASE_URL`` is ``https://dtc.aparavi.com/``; the SDK will
    automatically append the appropriate version prefix when calling the API.
    """
    if AparaviClient is None:
        print(
            "aparavi_dtc_sdk is not installed. Install it with 'pip install aparavi-dtc-sdk'"
            " and retry."
        )
        return

    # Default to the root SaaS domain. Do NOT include '/api/v1' here – the
    # SDK handles the version prefix internally.
    base_url = os.environ.get("DTC_BASE_URL", "https://dtc.aparavi.com/")
    api_key = os.environ.get("DTC_API_KEY")
    if not api_key:
        print(
            "Environment variable DTC_API_KEY is not set. Please provide your Aparavi"
            " API key via the DTC_API_KEY environment variable."
        )
        return

    # Instantiate the client
    client = AparaviClient(base_url, api_key)
    pipeline_config = load_pipeline_config()

    # Ensure sample_docs directory exists and has files to ingest
    sample_dir = Path(__file__).parent / "sample_docs"
    if not sample_dir.exists() or not any(sample_dir.iterdir()):
        print(
            "No sample_docs directory found or it is empty. Create it and add documents to ingest."
        )
        return

    # Create a unique task name
    task_name = f"ingest_{int(time.time())}"
    print(f"Executing ingestion pipeline '{task_name}'…")

    # Use the one-step workflow to execute the pipeline and upload files.
    # The file_glob matches all files within sample_docs (recursively).
    file_glob = str(sample_dir / "**" / "*")
    try:
        result = client.execute_pipeline_workflow(
            pipeline=pipeline_config,
            file_glob=file_glob,
            task_name=task_name,
        )
        # If the workflow completes immediately, print the result.
        print("Ingestion pipeline finished. Result:")
        print(result)
    except Exception as exc:
        print("Error running ingestion pipeline:", exc)

def ask_question(question: str) -> None:
    """
    Run a query pipeline to retrieve context and ask an LLM a question.

    This function builds a small query pipeline programmatically.  It embeds
    the user's question, performs a vector store retrieval from a Qdrant
    instance, and sends the retrieved context to a language model.  The
    function requires the following environment variables:

    - ``DTC_API_KEY`` and optionally ``DTC_BASE_URL`` for the Aparavi API.
    - ``QDRANT_HOST`` and ``QDRANT_API_KEY`` for your vector store.
    - ``OPENAI_API_KEY`` (or the key for another supported LLM provider).
    """
    if AparaviClient is None:
        print(
            "aparavi_dtc_sdk is not installed. Install it with 'pip install aparavi-dtc-sdk'"
            " and retry."
        )
        return
    # Use root domain for base_url; omit '/api/v1' because the SDK appends the version
    base_url = os.environ.get("DTC_BASE_URL", "https://dtc.aparavi.com/")
    api_key = os.environ.get("DTC_API_KEY")
    if not api_key:
        print(
            "Environment variable DTC_API_KEY is not set. Please provide your Aparavi"
            " API key via the DTC_API_KEY environment variable."
        )
        return
    client = AparaviClient(base_url, api_key)
    # Construct a minimal query pipeline
    pipeline: dict[str, object] = {
        "pipeline": {
            "components": [
                {
                    "type": "source",
                    "config": {"source_type": "text", "text": question},
                },
                {
                    "type": "embedding",
                    "config": {"model": "openai-embedding-ada-2"},
                },
                {
                    "type": "vector_store_retrieval",
                    "config": {
                        "store_type": "qdrant",
                        "k": 5,
                        "connection": {
                            "host": os.environ.get("QDRANT_HOST"),
                            "api_key": os.environ.get("QDRANT_API_KEY"),
                        },
                    },
                },
                {
                    "type": "llm",
                    "config": {
                        "model": "openai-chat-gpt-3.5-turbo",
                        "prompt_template": (
                            "You are an assistant answering questions based on the following context:\n"
                            "{context}\n\nQuestion: {query}\nAnswer:"
                        ),
                        "api_key": os.environ.get("OPENAI_API_KEY"),
                    },
                },
            ]
        }
    }
    print("Executing query pipeline…")
    try:
        # Use execute_pipeline_workflow for simplicity; no file_glob needed
        result = client.execute_pipeline_workflow(
            pipeline=pipeline,
            file_glob=None,
            task_name=f"query_{int(time.time())}",
        )
        # When the pipeline completes, result should contain the final output
        print("Query pipeline finished. Result:")
        print(result)
    except Exception as exc:
        print("Error executing query pipeline:", exc)


def test_project() -> None:
    """
    Test the ingestion and query pipeline end-to-end with a sample document.

    This helper function creates a simple document in ``sample_docs/`` (if the
    directory doesn’t exist), runs the ingestion pipeline, and then asks a
    question against the newly ingested data.  This is useful for verifying
    that your environment variables and pipeline configuration are set up
    correctly.
    """
    # Ensure sample_docs exists
    sample_dir = Path(__file__).parent / "sample_docs"
    sample_dir.mkdir(exist_ok=True)
    # Write a simple test document
    sample_file = sample_dir / "test_doc.txt"
    sample_file.write_text(
        "Aparavi Data Toolchain enables scalable document processing.",
        encoding="utf-8",
    )
    print("Sample document created at", sample_file)
    ingest_documents()
    ask_question("What does Aparavi Data Toolchain do?")


def check_env() -> None:
    """
    Print required environment variables for verification.

    Use this helper to quickly see which of the necessary environment variables
    are set in your current shell.  This can help diagnose issues when the
    ingestion or query functions complain about missing configuration.
    """
    print("DTC_API_KEY:", os.environ.get("DTC_API_KEY"))
    print("DTC_BASE_URL:", os.environ.get("DTC_BASE_URL"))
    print("QDRANT_HOST:", os.environ.get("QDRANT_HOST"))
    print("QDRANT_API_KEY:", os.environ.get("QDRANT_API_KEY"))
    print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo script for Aparavi DTC")
    parser.add_argument(
        "--ingest", action="store_true", help="Ingest documents using the pipeline config"
    )
    parser.add_argument(
        "--ask", type=str, help="Ask a question once documents have been ingested"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the whole project (ingest + ask)"
    )
    parser.add_argument(
        "--check-env", action="store_true", help="Print required environment variables"
    )
    args = parser.parse_args()
    if args.ingest:
        ingest_documents()
    elif args.ask:
        ask_question(args.ask)
    elif args.test:
        test_project()
    elif args.check_env:
        check_env()
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()