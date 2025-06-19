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

def discover_child_pages(page_id, discovered_pages=None):
    if discovered_pages is None:
        discovered_pages = set()
    if page_id in discovered_pages:
        return discovered_pages
    discovered_pages.add(page_id)
    try:
        response = notion.blocks.children.list(block_id=page_id)
        blocks = response.get("results", [])
        for block in blocks:
            block_type = block.get("type")
            if block_type == "child_page":
                child_page_id = block.get("id")
                if child_page_id and child_page_id not in discovered_pages:
                    discover_child_pages(child_page_id, discovered_pages)
            elif block_type == "child_database":
                # Optional: handle child_database if needed
                pass
    except Exception as e:
        print(f"⚠️ Error discovering children of page {page_id}: {e}")
    return discovered_pages

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
    all_page_ids = discover_child_pages(NOTION_PAGE_ID)
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