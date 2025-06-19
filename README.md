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

-   `--reasoner`  
    Use the more powerful (but slower) `deepseek-reasoner` model instead of the default `deepseek-chat`.
    ```bash
    python3 scripts/query_chroma.py --reasoner
    ```

-   `--model <model_name>`  
    Use a different DeepSeek model.
    ```bash
    python3 scripts/query_chroma.py --model deepseek-coder
    ```

### Customizing the AI's Personality

---
---
---

# 🐳 RAG Notion Tutorial (DeepSeek)

Let's build your project using **NotionPageReader + LlamaIndex + BAAI (Hugging Face) embeddings + Chroma + DeepSeek**.

## 1. Folder Structure and Working Environment

1. Create a folder for your project, for example:
    
    ```
    ~/projects/notion_llama_chroma/
    ```
    
2. Inside it, create the following subfolders to keep things organized:
    
    ```
    notion_llama_chroma/
    ├── scripts/        ← your Python files go here
    ├── data/           ← to store JSON files or local databases
    ├── chroma_db/      ← Chroma will store persistent vectors here
    ├── requirements.txt ← list of libraries to install
    └── README.md        ← general project explanation
    ```
    

> Why:
> 
> - Keeping **scripts/**, **data/**, and **chroma_db/** separate keeps the project clean.
> - **requirements.txt** helps track dependencies, especially if you want to reproduce the setup later.
> - **README.md** serves as a quick reference guide to jot down key steps.

---

## 2. Create and Activate a Virtual Environment (virtualenv)

To isolate dependencies from your global Python environment:

1. Open a terminal and navigate to your project folder:
    
    ```
    cd ~/projects/notion_llama_chroma
    ```
    
2. Create a virtual environment (Python 3.9+ recommended):
    
    ```
    python3 -m venv venv
    ```
    
3. Activate it:
    - On macOS/Linux:
        
        ```
        source venv/bin/activate
        ```
        
    - On Windows (PowerShell):
        
        ```
        venv\Scripts\Activate.ps1
        ```
        
4. Verify that the environment is active (you should see the prefix `(venv)` in your shell).

> Why:
> 
> 
> A virtual environment keeps this project's libraries isolated. This prevents conflicts if you later work on another project using different versions of `sentence-transformers` or `chromadb`.
> 

---

## 3. Create `requirements.txt` and Install Libraries

Inside your main project folder (`BookRAG/`), create a file named `requirements.txt` with the following contents:

```
openai>=1.0.0
llama-index>=0.9.0
llama-index-readers-notion>=0.1.0
llama-index-embeddings-huggingface>=0.1.0
llama-index-vector-stores-chroma>=0.1.0
chromadb>=0.4.0
python-dotenv>=1.0.0
notion-client>=2.0.0
requests>=2.31.0
tqdm>=4.66.0
tabulate>=0.9.0
rich>=13.7.0
```

### Explanation of Included Libraries

- `openai`: Required for compatibility with OpenAI-like SDKs (e.g., DeepSeek).
- `llama-index`: Core of LlamaIndex (formerly GPT Index).
- `llama-index-readers-notion`: Reader for Notion.
- `llama-index-embeddings-huggingface`: Enables use of Hugging Face embedding models like `BAAI/bge-large-en`.
- `chromadb`: Vector database with automatic persistence.
- `python-dotenv`: Loads environment variables from a `.env` file.
- `notion-client`: Official Notion API client, used to enrich metadata and discover subpages.
- `requests`: For HTTP calls used by various scripts and dependencies.
- `tqdm`: Progress bars in terminal (optional but helpful).
- `tabulate` and `rich`: Used to format tables and output in scripts.

> 💡 Note:
> 
> - If using Hugging Face models, `sentence-transformers` will be installed automatically as a dependency.
> - If using DeepSeek, ensure your `.env` file includes `OPENAI_API_KEY` and the correct API endpoint.

### Install All Dependencies

With your virtual environment active, run:

```
pip install -r requirements.txt
```

> ✅ Why use requirements.txt?
> 
> 
> It makes it easy to recreate the environment on any machine without manually installing each library.
> 

---

## 4. Notion Credentials (and Other Environment Variables)

Before you can read data from Notion, complete the following steps:

### 🛠️ Create a Notion Integration

1. Go to https://www.notion.so/my-integrations.
2. Create a new integration.
3. Grant it **read access**.
4. Copy the **Internal Integration Token** (your secret key).

### 🧾 Invite the Integration to Your Pages or Databases

1. In Notion, open the page or database you want to index.
2. Click **"Share"**.
3. Add your integration as a **guest with read permissions**.

### 📄 Create a `.env` File at the Project Root

To avoid typing your token every time, create a `.env` file in the root of your project (`BookRAG/`) with the following content:

```
NOTION_INTEGRATION_TOKEN=your_notion_token_here
NOTION_ROOT_PAGE_ID=your_root_page_id
```

If you're also using a model compatible with OpenAI (like DeepSeek), add your key and endpoint:

```
NOTION_INTEGRATION_TOKEN=secret_xxx
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.deepseek.com/v1
```

### 🚫 Add `.env` to `.gitignore` (Strongly Recommended)

Ensure your `.env` file is ignored by Git to avoid exposing secrets:

```
# .gitignore
.env
```

### ❓ Why Use a `.env` File?

- `python-dotenv` will automatically load the variables from `.env`.
- Your scripts (`notion_to_chroma.py`, `query_chroma.py`, etc.) will access the credentials without hardcoding them.
- **Never commit your `.env` file to public repos.** Protect your tokens.

---

## 5. First Script: `scripts/notion_to_chroma.py`

This script will:

- 🔍 Recursively discover all subpages from a Notion root page.
- 📖 Read the content of each discovered page.
- 🧠 Enrich documents with metadata: title, last edited time, and URL.
- 🧹 Filter out pages that haven't changed since last index.
- 🧬 Generate embeddings using `BAAI/bge-large-en`.
- 🧱 Initialize **ChromaDB** in local persistent mode.
- 📥 Index all new or modified documents.
- 🗂️ Update the indexing log in `indexed_pages.json`.

### 🧾 1. Add These Variables to `.env`

Adjust values to your project:

```
NOTION_INTEGRATION_TOKEN=secret_xxx
NOTION_ROOT_PAGE_ID=XXXXXXXXXXXXXXX
CHROMA_PERSIST_DIR=./chroma_storage
INDEXED_PAGES_LOG=./indexed_pages.json
EMBED_MODEL=BAAI/bge-large-en
```

### 🧠 2. Script File: `scripts/notion_to_chroma.py`

Save the script as `scripts/notion_to_chroma.py`. It will read values from `.env` using `python-dotenv`, so you don't need to pass arguments in the terminal.

```python
#!/usr/bin/env python3
"""
scripts/notion_to_chroma.py - Ingest Notion pages (and sub-pages) into ChromaDB for RAG pipelines.
Configurable via .env variables and/or command-line arguments.
"""
import os
import json
import argparse
import shutil
from dotenv import load_dotenv
from notion_client import Client as NotionClient

# Load environment variables
load_dotenv()
DEFAULT_NOTION_TOKEN = os.getenv("NOTION_INTEGRATION_TOKEN")
DEFAULT_NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DEFAULT_HISTORY_PATH = os.getenv("CHROMA_HISTORY_PATH", "./data/indexed_pages.json")

# Command-line arguments
parser = argparse.ArgumentParser(description="Ingest Notion into ChromaDB for RAG.")
parser.add_argument("--notion-token", default=DEFAULT_NOTION_TOKEN, help="Notion integration token.")
parser.add_argument("--notion-page-id", default=DEFAULT_NOTION_PAGE_ID, help="Root Notion page ID.")
parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="ChromaDB persistence directory.")
parser.add_argument("--history-path", default=DEFAULT_HISTORY_PATH, help="Indexing history JSON path.")
parser.add_argument("--force-reindex", action="store_true", help="Force reindexing even if no changes detected.")
parser.add_argument("--reset-chroma", action="store_true", help="Delete existing ChromaDB database and history before running.")
args = parser.parse_args()

NOTION_TOKEN = args.notion_token
NOTION_PAGE_ID = args.notion_page_id
PERSIST_DIR = args.persist_dir
HISTORY_PATH = args.history_path
FORCE_REINDEX = args.force_reindex

if NOTION_TOKEN is None:
    raise ValueError("❌ Missing NOTION_INTEGRATION_TOKEN in .env or --notion-token argument")
if NOTION_PAGE_ID is None:
    raise ValueError("❌ Missing NOTION_PAGE_ID in .env or --notion-page-id argument")

notion = NotionClient(auth=NOTION_TOKEN)

from llama_index.readers.notion import NotionPageReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings, DEFAULT_TENANT, DEFAULT_DATABASE
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

def load_history(history_path):
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(f"⚠️ The history file {history_path} is empty or corrupted. It will be reset.")
        return {}

def save_history(history, history_path):
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def should_reindex(doc, history):
    page_id = doc.metadata.get("page_id")
    last_edit = doc.metadata.get("last_edited_time")
    if page_id is None or last_edit is None:
        return True
    return (page_id not in history) or (history[page_id] != last_edit)

def _discover_child_pages_recursive(block_id: str, discovered_pages: set):
    """
    Recursively finds all child pages starting from a given block ID.
    Adds found page IDs to the provided `discovered_pages` set.
    This function explores any block, but only adds IDs of type 'child_page'.
    """
    try:
        next_cursor = None
        while True:
            response = notion.blocks.children.list(block_id=block_id, start_cursor=next_cursor, page_size=100)
            blocks = response.get("results", [])
            for block in blocks:
                # If it's a child page, add it and recurse into it.
                if block.get("type") == "child_page":
                    child_page_id = block.get("id")
                    if child_page_id not in discovered_pages:
                        discovered_pages.add(child_page_id)
                        _discover_child_pages_recursive(child_page_id, discovered_pages)
                # If it's any other block that has children, just recurse into it.
                elif block.get("has_children"):
                    _discover_child_pages_recursive(block.get("id"), discovered_pages)
            
            next_cursor = response.get("next_cursor")
            if not next_cursor:
                break
    except Exception as e:
        print(f"⚠️ Error exploring children of block {block_id}: {e}")

def enrich_document_metadata(doc, notion_client):
    pid = doc.metadata.get("page_id")
    if pid is None:
        return doc
    try:
        page_info = notion_client.pages.retrieve(page_id=pid)
    except Exception as e:
        print(f"⚠️ Error retrieving metadata for {pid}: {e}")
        return doc
    title = None
    props = page_info.get("properties", {})
    if "title" in props and props["title"].get("title"):
        title_list = props["title"]["title"]
        if title_list:
            title = title_list[0].get("plain_text")
    last_edited = page_info.get("last_edited_time")
    page_url = page_info.get("url")
    doc.metadata["page_id"] = pid
    if last_edited:
        doc.metadata["last_edited_time"] = last_edited
    if title:
        doc.metadata["title"] = title
    if page_url:
        doc.metadata["url"] = page_url
    return doc

def main():
    if args.reset_chroma:
        print("🔥 --reset-chroma flag detected. Deleting database and history.")
        if os.path.exists(PERSIST_DIR):
            try:
                shutil.rmtree(PERSIST_DIR)
                print(f"✅ Deleted ChromaDB directory: {PERSIST_DIR}")
            except OSError as e:
                print(f"❌ Error deleting directory {PERSIST_DIR}: {e}")
        else:
            print(f"🤷 Directory {PERSIST_DIR} not found, nothing to delete.")
        
        history_dir = os.path.dirname(HISTORY_PATH)
        if os.path.exists(history_dir):
            try:
                shutil.rmtree(history_dir)
                print(f"✅ Deleted history directory: {history_dir}")
            except OSError as e:
                print(f"❌ Error deleting directory {history_dir}: {e}")
        else:
            print(f"🤷 Directory {history_dir} not found, nothing to delete.")

        print("✅ Reset complete. Proceeding with fresh indexing...")

    print(f"🔗 Notion page ID: {NOTION_PAGE_ID}")
    print(f"💾 ChromaDB dir: {PERSIST_DIR}")
    print(f"📝 History: {HISTORY_PATH}")
    if FORCE_REINDEX:
        print("🔄 Forced reindexing enabled")

    history = load_history(HISTORY_PATH)

    # 1) Discover all pages and subpages
    print("🔍 Discovering pages and subpages...")
    all_page_ids = {NOTION_PAGE_ID}
    _discover_child_pages_recursive(NOTION_PAGE_ID, all_page_ids)
    print(f"📄 Found {len(all_page_ids)} pages (including subpages)")

    # 2) Read Notion documents
    reader = NotionPageReader(integration_token=NOTION_TOKEN)
    print("📖 Reading Notion pages...")
    docs = reader.load_data(page_ids=list(all_page_ids))
    print(f"✅ Loaded {len(docs)} Notion documents.")

    # 3) Enrich metadata
    enriched_docs = [enrich_document_metadata(doc, notion) for doc in docs]
    print(f"✅ Enriched metadata for {len(enriched_docs)} documents.")

    # 5) Embedding model
    print("🔄 Loading embedding model BAAI/bge-large-en...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")

    # 6) Persistent Chroma client
    print(f"💾 Initializing ChromaDB at {PERSIST_DIR}...")
    chroma_client = PersistentClient(
        path=PERSIST_DIR,
        settings=ChromaSettings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    chroma_collection = chroma_client.get_or_create_collection("notion_collection")
    
    # Check if the database is empty, which should trigger a re-index.
    is_db_empty = chroma_collection.count() == 0
    if is_db_empty:
        print("⚠️ ChromaDB is empty. Forcing a re-index of all documents.")

    # 4) Filter docs to index
    if FORCE_REINDEX or is_db_empty:
        docs_to_index = enriched_docs
        if FORCE_REINDEX:
            print(f"🔄 Reindexing {len(docs_to_index)} documents (forced by flag)")
        else:
            print(f"🔄 Reindexing {len(docs_to_index)} documents (database was empty)")
    else:
        docs_to_index = [doc for doc in enriched_docs if should_reindex(doc, history)]
        print(f"🆕 Documents to index based on last edit time: {len(docs_to_index)}")

    # We need the vector_store and storage_context after filtering
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 7) Index or ensure the index
    print("📚 Creating/updating index in ChromaDB...")
    if docs_to_index:
        index = VectorStoreIndex.from_documents(
            docs_to_index,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        print(f"✅ Indexed {len(docs_to_index)} new/modified documents.")
    else:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
        print("✅ Existing documents available in the index.")

    # 8) Persist changes
    index.storage_context.persist()
    print("✅ Index saved to ChromaDB.")

    # 9) Update history
    for doc in docs_to_index:
        pid = doc.metadata.get("page_id")
        last_edit = doc.metadata.get("last_edited_time")
        if pid and last_edit:
            history[pid] = last_edit
    if docs_to_index:
        save_history(history, HISTORY_PATH)
        print(f"📝 History updated ({len(docs_to_index)} pages indexed).")
    else:
        print("📝 No new pages to update in history.")

    # 10) Check documents in ChromaDB
    count = chroma_collection.count()
    print(f"📊 Total documents in ChromaDB: {count}")

if __name__ == "__main__":
    main()
```

---

## 6. How to Run Your First Script

1. Make sure your virtual environment is active (`(venv)` should appear in your shell).
2. Confirm that you have:
    - Created `requirements.txt` and installed packages (`pip install -r requirements.txt`).
    - A `.env` file with `NOTION_INTEGRATION_TOKEN=…`.
    - `data/` and `chroma_db/` folders (Chroma will create them automatically the first time).
3. From the root of your project, run:
    
    ```
    python3 scripts/notion_to_chroma.py
    ```
    
    You should see messages like:
    
    ```
    🔄 Loading embedding model BAAI/bge-large...
    💾 Initializing Chroma at: ../chroma_db
    📚 Creating/updating index in Chroma...
    ✅ Index saved to Chroma.
    📝 History updated (X pages indexed).
    ```
    
4. If you run the script again without modifying any Notion pages, you should see:
    
    ```
    ✅ No new or modified pages to reindex.
    ```
    

---

## 7. What's Next After This First Run?

1. **Semantic Querying**
    
    Once your index is stored in Chroma, you can ask RAG (Retrieval-Augmented Generation) questions. For example, create a script `scripts/query_chroma.py` with the following content:
    

```python
#!/usr/bin/env python3
"""
scripts/query_chroma.py - Interactive RAG chat with streaming responses from DeepSeek.

This script connects to the existing ChromaDB, retrieves relevant documents,
and uses the DeepSeek API to generate responses in an interactive chat session.
"""

import os
import argparse
import time
from dotenv import load_dotenv

# --- LlamaIndex Imports ---
# Use the correct imports for the installed version
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI as BaseLlamaOpenAI
from llama_index.core.memory import ChatMemoryBuffer

# --- ChromaDB Imports ---
import chromadb

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_storage")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-large-en")

class DeepseekOpenAI(BaseLlamaOpenAI):
    """Subclass to bypass model validation for DeepSeek models."""
    def _get_model_name(self) -> str:
        # This is a workaround to use DeepSeek with the OpenAI class.
        # The underlying API calls will still go to the DeepSeek endpoint.
        return "gpt-3.5-turbo"

def print_streaming_response(response_gen):
    """Prints a streaming response from the chat engine token by token."""
    print("🤖 Assistant: ", end="", flush=True)
    full_response = ""
    
    try:
        # The response object from `stream_chat` is a StreamingAgentChatResponse
        if not hasattr(response_gen, 'response_gen') or response_gen.response_gen is None:
            print("[No streamable response generated]")
            return ""

        for chunk in response_gen.response_gen:
            token = ""
            # The structure of the chunk can vary. We need to handle different possibilities.
            if isinstance(chunk, str):
                token = chunk
            elif hasattr(chunk, 'delta') and chunk.delta:
                token = chunk.delta
            elif hasattr(chunk, 'text') and chunk.text:
                token = chunk.text
            elif hasattr(chunk, 'content') and chunk.content:
                token = chunk.content
            
            if token:
                print(token, end="", flush=True)
                full_response += token
                time.sleep(0.01)  # Small delay for a more natural streaming effect

    except KeyboardInterrupt:
        print("\n\n⚠️ Response interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during streaming: {e}")
    finally:
        print()  # Ensure a new line after the response
    
    if not full_response and "Error" not in full_response:
        print("[No response generated]")
        
    return full_response

def main():
    """Main function to run the interactive RAG chat."""
    parser = argparse.ArgumentParser(description="Interactive RAG chat with DeepSeek.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top similar documents to retrieve.")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model to use (e.g., deepseek-chat, deepseek-coder).")
    parser.add_argument("--persist-dir", default=CHROMA_PERSIST_DIR, help="Directory for ChromaDB persistence.")
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        raise ValueError("❌ DEEPSEEK_API_KEY not found in .env file. Please add it.")

    print(f"🚀 Starting RAG chat with model: {args.model}")
    print(f"💾 Using ChromaDB from: {args.persist_dir}")
    print(f"🔍 Retrieving top {args.top_k} documents for context.")

    # 1. Initialize LLM
    llm = DeepseekOpenAI(
        api_key=DEEPSEEK_API_KEY,
        api_base="https://api.deepseek.com",
        model=args.model,
        temperature=0.1,  # Slightly creative, but not too random
    )

    # 2. Initialize Embedding Model
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    # 3. Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=args.persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("notion_collection")
    
    if chroma_collection.count() == 0:
        print("⚠️ ChromaDB is empty. Run the ingestion script first.")
        return

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 4. Load index from vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # 5. Create chat engine with memory
    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
    
    chat_engine = index.as_chat_engine(
        llm=llm,
        similarity_top_k=args.top_k,
        memory=memory,
        streaming=True,
        verbose=False,
    )

    print("-" * 50)
    print("💬 Ask questions about your Notion documents. Type 'exit' or 'quit' to end.")

    try:
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                response_stream = chat_engine.stream_chat(user_input)
                print_streaming_response(response_stream)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 
```

- **Run**:
    
    ```
    python3 scripts/query_chroma.py
    ```
    
- You'll see the prompt `Assistant:`; type your question and you'll get a response enriched by your Notion index.