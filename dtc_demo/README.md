# Document Q&A Demo Using Aparavi Data Toolchain

This repository contains a minimal demonstration of how to use the **Aparavi Data Toolchain for AI (DTC)** to build a retrieval‑augmented question answering system over unstructured documents.

## Use case

Imagine an enterprise that stores technical manuals, policy documents and meeting notes across disparate file systems (NAS, SharePoint, cloud drives). Employees often need quick answers to questions such as *“What are the leave policies?”* or *“How do I perform a secure code review?”*.  Instead of manually searching through hundreds of files, this demo shows how the Aparavi Data Toolchain can ingest, classify and vectorize documents so that a large language model (LLM) can retrieve relevant information and answer questions.

The demo assumes a set of local PDF and text files located in a directory. The pipeline uses DTC to:

- **Ingest** files from the local file system via the DTC `source` node.
- **Index and classify** documents to extract metadata and detect sensitive content.
- **Chunk and embed** the text into vector representations with an embedding model (e.g. OpenAI `text-embedding-ada-002`).
- **Store** embeddings in a vector database (Qdrant in this example).
- **Query** the vector store with a user question, retrieve the most relevant chunks, and call an LLM to generate an answer.

By automating this workflow, enterprises can quickly build a self‑service Q&A assistant for employees, improving productivity and reducing time spent sifting through documents.

## Repository contents

- `app.py` – A Python script using the [dtc‑api‑sdk](https://pypi.org/project/aparavi-dtc-sdk/) to define, validate and execute DTC pipelines for ingestion and querying.  The script demonstrates both the ingestion pipeline (to scan, index, chunk, embed and store documents) and a simple query pipeline to answer questions.
- `pipeline_config.json` – A JSON template showing the structure of the ingestion pipeline, which can be edited and passed to the API from `app.py`.
- `sample_docs/` – A folder where you can place your own PDF/TXT/Word documents for ingestion.  (This directory is not included in this template; create it yourself and populate it with sample files.)
  
  This repository includes several example `.txt` files under `sample_docs/` (e.g. parental leave, reimbursement and security policies) so that you can run the demo immediately.  Feel free to replace or augment these with your own documents.

- `dtc_local_demo.py` – A local pipeline simulator that mirrors the DTC workflow without requiring any API access.  It uses scikit‑learn’s TF‑IDF vectoriser and nearest‑neighbour index to embed and retrieve text chunks from your documents.  See the **Running the local demo (no API keys required)** section below for details.

## How to run

1. **Install dependencies**

   ```bash
   pip install aparavi-dtc-sdk qdrant-client openai
   ```

   You will also need API keys for Aparavi DTC (Community Edition or Enterprise), Qdrant and the chosen LLM provider (e.g. OpenAI).  These can be provided as environment variables or configured in the script.

2. **Prepare documents**

   Create a directory called `sample_docs/` inside `dtc_demo/` and copy the documents you wish to process.  Ensure the files are text‑based or scanned images (the pipeline includes OCR support via DTC).

3. **Run ingestion**

   Execute the ingestion pipeline defined in `app.py` to scan, index, chunk and embed documents and store the vectors in Qdrant:

   ```bash
   python app.py --ingest
   ```

   The script will upload your documents to the DTC ingestion endpoint, wait for the pipeline to complete and verify that vectors have been stored.

4. **Ask questions**

   After the ingestion pipeline finishes, run the query function to embed a question, retrieve the most relevant chunks, and call the LLM for an answer:

   ```bash
   python app.py --ask "What are the leave policies in the employee handbook?"
   ```

   The script prints the retrieved context and the generated answer.

## Running the local demo (no API keys required)

If you are using the Community Edition and cannot obtain an API key, you can still run a fully offline demonstration of the ingestion and retrieval pipeline using the `dtc_local_demo.py` script.  This script requires only Python, NumPy and scikit‑learn (both are pre‑installed in most environments) and **does not call out to any external services**.

1. **Prepare documents** – The repository already contains a set of example `.txt` files in the `sample_docs/` folder.  You may add, remove or modify files in this directory to suit your demo.  Only plain text files are supported by the local pipeline.

2. **Ingest and build the index** – From the `dtc_demo/` directory, run:

   ```bash
   python dtc_local_demo.py --ingest
   ```

   This reads all `.txt` files in `sample_docs/`, splits each into ~200‑word chunks, fits a TF‑IDF vectoriser, builds a cosine‑similarity index over the resulting vectors and saves the model to `ingested_index.pkl` in the project directory.  It will report the number of documents and chunks processed.

3. **Ask questions** – Once the index is built, you can query it with a natural language question, for example:

   ```bash
   python dtc_local_demo.py --ask "What does the parental leave policy cover?"
   ```

   The script embeds your question, retrieves the most similar chunks from the index and prints the relevant passages along with the file names and chunk numbers.  This demonstrates the retrieval step of a RAG pipeline without invoking an LLM.

4. **Run a combined test** – To ingest and immediately run a demo query in one command, use:

   ```bash
   python dtc_local_demo.py --test
   ```

   This will rebuild the index (overwriting any existing `ingested_index.pkl`) and then run a built‑in example question.  It's a quick way to verify that everything is working end‑to‑end.

The local demo is a simplified simulation of the DTC pipeline.  It does not perform OCR, classification or embedding with deep models, but it captures the core idea of turning unstructured documents into searchable vector representations and answering questions by retrieving relevant context.

## Design decisions

The demo aims to highlight DTC’s strengths without building a full‑stack application:

- **Low‑code pipeline definition.**  By storing the pipeline JSON in `pipeline_config.json`, we can visualise and modify it using the DTC UI while still automating execution via the SDK.
- **Local file ingestion.**  We chose the local file system connector for simplicity; other connectors (S3, Google Drive, SharePoint) can be substituted by changing the `source_type` and authentication details.
- **Chunking and embedding.**  Documents are split into 500‑token chunks with 50‑token overlap, which provides a balance between context size and retrieval granularity.  Embeddings use the `text-embedding-ada-002` model because it is widely available and cost effective.
- **Vector storage with Qdrant.**  Qdrant is an open‑source vector database that integrates smoothly with DTC.  You can replace this with Milvus, Pinecone or another supported store.
- **Command‑line interface.**  A simple CLI demonstrates how to run ingestion and querying; a future extension could include a web UI or chat interface.

## Feedback on the Aparavi Data Toolchain

During development we found several aspects of DTC particularly valuable:

- **Comprehensive connectors and processing options.**  The ability to ingest from numerous data sources, run OCR, detect sensitive data, and handle unstructured formats reduces the need for bespoke ETL code【542602203600026†screenshot】.
- **Visual workflow builder and SDK synergy.**  Building pipelines in the UI and then exporting them as JSON for automation allowed for quick iteration and reproducibility.【542602203600026†screenshot】
- **Integration with multiple vector databases and LLMs.**  The platform abstracts away the details of embedding and storage, making it easy to swap technologies.

However, there were also challenges and areas for improvement:

- **Documentation clarity.**  While the core concepts are documented, some advanced features (such as custom chunking strategies and API error handling) lacked detailed examples.
- **Community edition limits.**  The 5 GB / 5,000 file limit of the free tier restricts large prototypes.  A more generous trial would encourage deeper experimentation.
- **SDK ergonomics.**  The Python SDK requires manual polling and uploading of files; higher‑level helper functions (e.g. to monitor pipeline status and stream logs) would improve the developer experience.

## Contributing

This repository is intended as a self‑contained demo and is not affiliated with Aparavi.  Feel free to fork and adapt it for your own experiments.  For questions about the Data Toolchain itself, refer to the official [Aparavi documentation](https://aparavi.com/documentation-aparavi/).
