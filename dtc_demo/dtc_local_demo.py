"""
Local DTC demo pipeline
=======================

This script provides a lightweight, local alternative to the Aparavi Data Toolchain
for demonstration purposes.  It implements a simplified ingestion and retrieval
pipeline using only open‑source Python libraries, so no API keys or external
services are required.

The pipeline mirrors the high‑level flow of the DTC:

1. **Ingestion** – Walk through a folder of unstructured text files and read
   their contents.  Each document is broken into smaller text chunks to improve
   retrieval granularity.
2. **Vectorisation** – Convert text chunks into numerical vectors using a
   `TfidfVectorizer` from scikit‑learn.  While not as semantically rich as
   transformer embeddings, TF‑IDF vectors are fast to compute and work
   offline.
3. **Indexing** – Build a nearest‑neighbour index on the vectorised chunks
   using `NearestNeighbors` with cosine similarity.  The index is persisted
   alongside the vectoriser for later use.
4. **Querying** – Embed a user’s question with the same vectoriser and
   retrieve the most similar chunks from the index.  The top results are
   returned as context for answering the query.

You can test the pipeline end‑to‑end by placing `.txt` files in the
`sample_docs/` directory and running this script with `--ingest` followed by
`--ask "your question"`.  The ingestion step builds and saves the index to
`ingested_index.pkl`.  The ask step loads the index and prints the most
relevant contexts for your query.  A combined `--test` command runs both
steps sequentially.

Note: Because this implementation uses TF‑IDF vectors, it does not require
internet access or any pre‑trained models.  If you wish to use more
sophisticated embeddings, consider installing `sentence-transformers` and
substituting the vectorisation stage.

Usage:
  python dtc_local_demo.py --ingest
      Ingest all .txt files in sample_docs/ and build the index.
  python dtc_local_demo.py --ask "Your question here"
      Query the index with a question (after ingesting).
  python dtc_local_demo.py --test
      Ingest and run a sample query.
  python dtc_local_demo.py --test-complete
      Create a sample doc, ingest, and run a demo query end-to-end.

Requirements:
  pip install numpy scikit-learn
"""

# Requirements:
#   pip install numpy scikit-learn

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


@dataclass
class DocumentChunk:
    """A simple container for a document chunk."""

    doc_id: str
    chunk_index: int
    text: str

    def __repr__(self) -> str:
        return f"{self.doc_id}[chunk {self.chunk_index}]"


@dataclass
class IngestionResult:
    """Result of an ingestion run, including metadata and the index."""

    vectorizer: TfidfVectorizer
    index: NearestNeighbors
    chunks: List[DocumentChunk]

    def save(self, path: Path) -> None:
        """Persist the ingestion result to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "IngestionResult":
        with open(path, "rb") as f:
            return pickle.load(f)


def read_documents(root_dir: Path) -> List[Tuple[str, str]]:
    """Recursively read all `.txt` files in `root_dir`.

    Returns a list of `(doc_id, text)` pairs, where `doc_id` is the
    relative path of the document (without extension).
    """
    documents: List[Tuple[str, str]] = []
    for file_path in root_dir.rglob("*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            text = file_path.read_text(encoding="latin-1")
        doc_id = str(file_path.relative_to(root_dir))
        documents.append((doc_id, text))
    return documents


def chunk_text(text: str, tokens_per_chunk: int = 200) -> List[str]:
    """Split a long string into chunks of roughly `tokens_per_chunk` words."""
    words = text.split()
    chunks = [" ".join(words[i : i + tokens_per_chunk]) for i in range(0, len(words), tokens_per_chunk)]
    return [chunk for chunk in chunks if chunk.strip()]


def ingest(root_dir: Path, index_path: Path) -> IngestionResult:
    """Ingest documents from `root_dir` and build a vector index at `index_path`.

    The function reads all text files, chunks them, fits a TF‑IDF vectoriser,
    builds a nearest‑neighbour index on the resulting vectors, and persists
    everything to disk.  It returns the in‑memory result as an `IngestionResult`.
    """
    documents = read_documents(root_dir)
    if not documents:
        raise ValueError(f"No text files found in {root_dir}")
    # Prepare chunks with identifiers
    chunks: List[DocumentChunk] = []
    for doc_id, text in documents:
        for i, chunk_text_str in enumerate(chunk_text(text)):
            chunks.append(DocumentChunk(doc_id=doc_id, chunk_index=i, text=chunk_text_str))
    # Extract the raw texts for vectorisation
    raw_texts = [chunk.text for chunk in chunks]
    # Fit the vectoriser on all chunk texts
    vectorizer = TfidfVectorizer(stop_words="english")
    embeddings = vectorizer.fit_transform(raw_texts)
    # Build nearest neighbour index
    index = NearestNeighbors(n_neighbors=5, metric="cosine")
    index.fit(embeddings)
    result = IngestionResult(vectorizer=vectorizer, index=index, chunks=chunks)
    result.save(index_path)
    return result


def query(index_path: Path, question: str, top_k: int = 3) -> List[str]:
    """Query the ingested index with a question and return the top contexts.

    Loads the saved index and vectoriser from `index_path`, embeds the
    question, performs a nearest‑neighbour search, and returns the texts of
    the most similar chunks.
    """
    ingestion = IngestionResult.load(index_path)
    vectorizer = ingestion.vectorizer
    index = ingestion.index
    chunks = ingestion.chunks
    # Embed the question
    question_vec = vectorizer.transform([question])
    # Query the index
    distances, indices = index.kneighbors(question_vec, n_neighbors=top_k)
    # Flatten indices and return associated texts
    results = []
    for idx in indices[0]:
        chunk = chunks[idx]
        snippet = chunk.text
        header = f"From {chunk.doc_id} (chunk {chunk.chunk_index}):"
        results.append(f"{header}\n{snippet}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Local DTC demo pipeline")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest all .txt documents in sample_docs and build the vector index",
    )
    parser.add_argument(
        "--ask", type=str, help="Ask a question after the index has been built"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run ingestion followed by a sample query to verify the pipeline",
    )
    parser.add_argument(
        "--test-complete",
        action="store_true",
        help="Create a sample doc, ingest, and run a demo query end-to-end",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).parent / "sample_docs"
    index_path = Path(__file__).parent / "ingested_index.pkl"

    if args.ingest or args.test:
        print(f"Ingesting documents from {root_dir}…")
        result = ingest(root_dir, index_path)
        print(f"Ingested {len(result.chunks)} chunks from {len(set(c.doc_id for c in result.chunks))} documents.")
        print(f"Index saved to {index_path}.")

    if args.ask is not None:
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index file {index_path} does not exist. Run with --ingest first."
            )
        query_str = args.ask
        print(f"Query: {query_str}")
        contexts = query(index_path, query_str, top_k=3)
        print("\nRelevant contexts:")
        for i, ctx in enumerate(contexts, 1):
            print(f"\n---- Result {i} ----\n{ctx}\n")

    if args.test:
        # Ask a demo question
        demo_q = "What does the parental leave policy cover?"
        print(f"\nRunning a demo query: {demo_q}")
        contexts = query(index_path, demo_q, top_k=3)
        print("\nRelevant contexts:")
        for i, ctx in enumerate(contexts, 1):
            print(f"\n---- Result {i} ----\n{ctx}\n")

    if args.test_complete:
        # Ensure sample_docs exists and has at least one file
        root_dir.mkdir(exist_ok=True)
        sample_file = root_dir / "test_doc.txt"
        if not any(root_dir.glob("*.txt")):
            sample_file.write_text(
                "The parental leave policy covers paid time off for new parents, including maternity, paternity, and adoption leave. Employees are eligible for up to 12 weeks of paid leave.",
                encoding="utf-8",
            )
            print(f"Sample document created at {sample_file}")
        print(f"Ingesting documents from {root_dir}…")
        result = ingest(root_dir, index_path)
        print(f"Ingested {len(result.chunks)} chunks from {len(set(c.doc_id for c in result.chunks))} documents.")
        print(f"Index saved to {index_path}.")
        demo_q = "What does the parental leave policy cover?"
        print(f"\nRunning a demo query: {demo_q}")
        contexts = query(index_path, demo_q, top_k=3)
        print("\nRelevant contexts:")
        for i, ctx in enumerate(contexts, 1):
            print(f"\n---- Result {i} ----\n{ctx}\n")


if __name__ == "__main__":  # pragma: no cover
    main()