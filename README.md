# RAG Notion Project

A simple project to ingest Notion pages into a ChromaDB vector store for RAG pipelines.

## 1. Setup

First time setup to prepare the environment.

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install required packages
pip install -r requirements.txt

# 3. Configure environment variables
# Copy the example file and fill in your tokens and IDs
cp env.example .env
```

## 2. Indexing Notion Pages

Use the script `scripts/notion_to_chroma.py` to pull data from Notion and store it in ChromaDB.

### Basic Command

This command will index new or modified pages based on its history log.

```bash
python3 scripts/notion_to_chroma.py
```

### Available Flags

You can modify the script's behavior with 자랑:

-   `--force-reindex`  
    Forces the script to re-index all Notion pages, ignoring the history log.
    ```bash
    python3 scripts/notion_to_chroma.py --force-reindex
    ```

-   `--reset-chroma`  
    **Deletes the database and history.** Use this for a complete, clean start.
    ```bash
    python3 scripts/notion_to_chroma.py --reset-chroma
    ```

-   `--notion-page-id YOUR_PAGE_ID`  
    Temporarily use a different root Notion page without changing the `.env` file.
    ```bash
    python3 scripts/notion_to_chroma.py --notion-page-id <ID_DE_PAGINA>
    ```

-   `--persist-dir path/to/db`  
    Specify a different location for the ChromaDB database.
    ```bash
    python3 scripts/notion_to_chroma.py --persist-dir ./custom_db
    ```

## 3. Querying the Database

Once your data is indexed, you can start an interactive chat session to ask questions using the `scripts/query_chroma.py` script.

### Basic Command

Run this command to start the chat. It will load your database and connect to the DeepSeek API.

```bash
python3 scripts/query_chroma.py
```

Inside the chat, you can type your questions. Use `exit` or `quit` to end the session.

### Available Flags

-   `--top-k <number>`  
    Change the number of documents retrieved from the database to build the answer.
    ```bash
    python3 scripts/query_chroma.py --top-k 5
    ```

-   `--model <model_name>`  
    Use a different DeepSeek model.
    ```bash
    python3 scripts/query_chroma.py --model deepseek-coder
    ```
