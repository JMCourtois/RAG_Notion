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
import chromadb

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
parser.add_argument("--sync", action="store_true", help="Force a sync and print debug info about deleted pages.")
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

def sync_deleted_pages(chroma_collection, history: dict, notion_page_ids: set):
    """
    Removes documents from ChromaDB and history if their source page no longer exists in Notion.
    This function performs a full audit of the database.
    """
    print("🔄 Syncing deleted pages...")
    
    # 1. Get ALL documents from ChromaDB to perform a full audit.
    try:
        all_chroma_docs = chroma_collection.get(include=["metadatas"])
        all_doc_ids = all_chroma_docs.get('ids', [])
        all_metadatas = all_chroma_docs.get('metadatas', [])
    except Exception as e:
        print(f"⚠️ Could not get documents from ChromaDB to sync deletes: {e}")
        return

    # 2. Identify document chunks whose page_id is no longer in Notion
    doc_ids_to_delete = []
    chroma_page_ids = set() # For history cleanup
    for i, meta in enumerate(all_metadatas):
        doc_page_id = meta.get('page_id')
        if doc_page_id:
            chroma_page_ids.add(doc_page_id)
            if doc_page_id not in notion_page_ids:
                doc_ids_to_delete.append(all_doc_ids[i])

    # 3. Delete the orphaned chunks from ChromaDB
    if doc_ids_to_delete:
        print(f"🗑️ Found {len(doc_ids_to_delete)} orphaned document chunks to delete...")
        try:
            chroma_collection.delete(ids=doc_ids_to_delete)
            print("✅ Successfully deleted orphaned chunks from ChromaDB.")
        except Exception as e:
            print(f"❌ Error deleting chunks from ChromaDB: {e}")
    else:
        print("✅ No orphaned document chunks found in ChromaDB.")

    # 4. Clean up the history file based on what's no longer in Notion
    history_page_ids_to_delete = set(history.keys()) - notion_page_ids
    if history_page_ids_to_delete:
        print(f"🗑️ Found {len(history_page_ids_to_delete)} page(s) to remove from history...")
        for page_id in history_page_ids_to_delete:
            del history[page_id]
        print("✅ Successfully removed pages from history.")
    else:
        print("✅ History file is already up to date.")

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

    # Connect to ChromaDB and load history before syncing
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection("notion_collection")

    # 2) Sync and remove deleted pages before doing anything else
    sync_deleted_pages(chroma_collection, history, all_page_ids)
    
    # 3) Read Notion documents for the remaining pages
    reader = NotionPageReader(integration_token=NOTION_TOKEN)
    print("📖 Reading Notion pages...")
    docs = reader.load_data(page_ids=list(all_page_ids))
    print(f"✅ Loaded {len(docs)} Notion documents.")

    # 4) Enrich metadata
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

    # 7) Filter docs to index
    if FORCE_REINDEX or is_db_empty:
        docs_to_index = enriched_docs
        if FORCE_REINDEX:
            print(f"🔄 Reindexing {len(docs_to_index)} documents (forced by flag)")
        else:
            print(f"🔄 Reindexing {len(docs_to_index)} documents (database was empty)")
    else:
        docs_to_index = [doc for doc in enriched_docs if should_reindex(doc, history)]
        print(f"🆕 Documents to index based on last edit time: {len(docs_to_index)}")

    # 8) Cleanly update the index by first deleting old chunks
    if not is_db_empty and docs_to_index:
        page_ids_to_update = {doc.metadata['page_id'] for doc in docs_to_index}
        print(f"🔄 Preparing to update {len(page_ids_to_update)} page(s) by deleting their old chunks first...")
        delete_filter = {"page_id": {"$in": list(page_ids_to_update)}}
        try:
            chroma_collection.delete(where=delete_filter)
            print("✅ Successfully deleted old chunks for pages to be updated.")
        except Exception as e:
            print(f"❌ Error deleting old chunks during update: {e}")

    # 9) Index or ensure the index
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

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

    # 10) Persist changes
    index.storage_context.persist()
    print("✅ Index saved to ChromaDB.")

    # 11) Update history
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

    # 12) Check documents in ChromaDB
    count = chroma_collection.count()
    print(f"📊 Total documents in ChromaDB: {count}")

if __name__ == "__main__":
    main()