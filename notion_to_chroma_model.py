#!/usr/bin/env python3
"""
notion_to_chroma.py - Ingest Notion pages into ChromaDB for RAG pipelines.

This script automatically ingests the configured Notion page and all its sub-pages into ChromaDB.
The page ID is hardcoded in the script for convenience.

Environment variables required in .env:
  NOTION_INTEGRATION_TOKEN=<your_notion_token>

Example:
  python notion_to_chroma.py
"""
import os
import json
import argparse
from dotenv import load_dotenv
from notion_client import Client as NotionClient

# Load environment variables
load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_INTEGRATION_TOKEN")
if NOTION_TOKEN is None:
    raise ValueError("❌ NOTION_INTEGRATION_TOKEN not found in .env. Please create a .env file with this variable.")

# =============================================================================
# CONFIGURATION: Set your Notion page ID here
# =============================================================================
# Replace this with your actual Notion page ID
# You can find the page ID in the URL: https://notion.so/your-page-name-20a881aecb0c8015ab5de6a618c00226
NOTION_PAGE_ID = "20a881aecb0c8015ab5de6a618c00226"
# =============================================================================

# Notion API client for metadata enrichment
notion = NotionClient(auth=NOTION_TOKEN)

from llama_index.readers.notion import NotionPageReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings, DEFAULT_TENANT, DEFAULT_DATABASE
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

def load_history(history_path):
    """Load indexing history from JSON file."""
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_history(history, history_path):
    """Save indexing history to JSON file."""
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def should_reindex(doc, history):
    """Check if document needs to be reindexed based on history."""
    page_id = doc.metadata.get("page_id")
    last_edit = doc.metadata.get("last_edited_time")
    if page_id is None or last_edit is None:
        return True
    return (page_id not in history) or (history[page_id] != last_edit)

def discover_child_pages(page_id, discovered_pages=None):
    """
    Recursively discover all child pages under a given page ID.
    Returns a set of all page IDs (including the parent and all children).
    """
    if discovered_pages is None:
        discovered_pages = set()
    
    if page_id in discovered_pages:
        return discovered_pages
    
    discovered_pages.add(page_id)
    print(f"🔍 Exploring page: {page_id}")
    
    try:
        # Get the page content to find child pages
        response = notion.blocks.children.list(block_id=page_id)
        blocks = response.get("results", [])
        print(f"   Found {len(blocks)} blocks in page {page_id}")
        
        for i, block in enumerate(blocks):
            block_type = block.get("type")
            block_id = block.get("id")
            print(f"   Block {i+1}: type={block_type}, id={block_id}")
            
            if block_type == "child_page":
                child_page_id = block.get("id")
                if child_page_id and child_page_id not in discovered_pages:
                    print(f"🔍 Found child page: {child_page_id}")
                    # Recursively discover children of this child page
                    discover_child_pages(child_page_id, discovered_pages)
            elif block_type == "child_database":
                database_id = block.get("id")
                print(f"🔍 Found child database: {database_id}")
                # Note: We could also process database pages if needed
                    
    except Exception as e:
        print(f"⚠️ Error discovering children for page {page_id}: {e}")
    
    return discovered_pages

def enrich_document_metadata(doc, notion_client):
    """Enrich a document with Notion metadata."""
    pid = doc.metadata.get("page_id")
    if pid is None:
        return doc
    
    try:
        page_info = notion_client.pages.retrieve(page_id=pid)
    except Exception as e:
        print(f"⚠️ Error retrieving metadata for {pid}: {e}")
        return doc
    
    # Extract fields
    title = None
    props = page_info.get("properties", {})
    if "title" in props and props["title"].get("title"):
        title_list = props["title"]["title"]
        if title_list:
            title = title_list[0].get("plain_text")
    last_edited = page_info.get("last_edited_time")
    page_url = page_info.get("url")
    
    # Inject metadata
    doc.metadata["page_id"] = pid
    if last_edited:
        doc.metadata["last_edited_time"] = last_edited
    if title:
        doc.metadata["title"] = title
    if page_url:
        doc.metadata["url"] = page_url
    
    return doc

def main():
    parser = argparse.ArgumentParser(description="Ingest configured Notion page and all sub-pages into ChromaDB for RAG.")
    parser.add_argument("--persist-dir", default="../chroma_db", help="Directory for ChromaDB persistence.")
    parser.add_argument("--history-path", default="../data/indexed_pages.json", help="Path for indexing history JSON.")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindexing even if no changes detected.")
    args = parser.parse_args()

    persist_dir = args.persist_dir
    history_path = args.history_path
    force_reindex = args.force_reindex

    print(f"🔗 Notion page ID to ingest: {NOTION_PAGE_ID}")
    print(f"💾 ChromaDB persistence dir: {persist_dir}")
    print(f"📝 Indexing history path: {history_path}")
    if force_reindex:
        print("🔄 Force reindexing enabled - will reindex even if no changes detected")

    history = load_history(history_path)

    # 1) Discover all child pages recursively
    print("🔍 Discovering all pages and sub-pages...")
    all_page_ids = discover_child_pages(NOTION_PAGE_ID)
    print(f"📄 Found {len(all_page_ids)} total pages (including sub-pages): {list(all_page_ids)}")

    # 2) Read all Notion docs (parent + all children)
    reader = NotionPageReader(integration_token=NOTION_TOKEN)
    print("📖 Reading all Notion pages...")
    docs = reader.load_data(page_ids=list(all_page_ids))
    print(f"✅ Loaded {len(docs)} documents from Notion.")

    # 3) Enrich docs with Notion metadata
    enriched_docs = []
    print("🔍 Enriching documents with Notion metadata...")
    for doc in docs:
        enriched_doc = enrich_document_metadata(doc, notion)
        enriched_docs.append(enriched_doc)

    print(f"✅ Enriched {len(enriched_docs)} documents.")
    for i, doc in enumerate(enriched_docs, 1):
        title = doc.metadata.get('title', 'No title')
        page_id = doc.metadata.get('page_id', 'No ID')
        last_edit = doc.metadata.get('last_edited_time', 'No timestamp')
        print(f"   {i}. {title} (ID: {page_id}) | last_edit: {last_edit}")

    # 4) Determine which docs to index
    if force_reindex:
        docs_to_index = enriched_docs
        print(f"🔄 Force reindexing: {len(docs_to_index)} documents will be reindexed")
    else:
        docs_to_index = [doc for doc in enriched_docs if should_reindex(doc, history)]
        print(f"🆕 Documents to (re)index: {len(docs_to_index)}")
    
    # 5) Load embedding model
    print("🔄 Loading embedding model BAAI/bge-large-en...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")

    # 6) Initialize persistent ChromaDB
    print(f"💾 Initializing ChromaDB at {persist_dir}...")
    chroma_client = PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    chroma_collection = chroma_client.get_or_create_collection("notion_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 7) Always create/update index in ChromaDB to ensure documents are available
    print("📚 Creating/updating index in ChromaDB...")
    if docs_to_index:
        # Index new/modified documents
        index = VectorStoreIndex.from_documents(
            docs_to_index,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        print(f"✅ Indexed {len(docs_to_index)} new/modified documents.")
    else:
        # Even if no new docs, ensure the index exists with existing documents
        print("📚 Ensuring existing documents are indexed...")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
        print("✅ Existing documents are available in the index.")
    
    # 8) Persist the storage context to ensure changes are saved
    index.storage_context.persist()
    print("✅ Index saved to ChromaDB.")

    # 9) Update history for indexed documents
    for doc in docs_to_index:
        pid = doc.metadata.get("page_id")
        last_edit = doc.metadata.get("last_edited_time")
        if pid and last_edit:
            history[pid] = last_edit
    
    if docs_to_index:
        save_history(history, history_path)
        print(f"📝 History updated ({len(docs_to_index)} pages indexed).")
    else:
        print("📝 No new pages to update in history.")

    # 10) Verify documents are available
    count = chroma_collection.count()
    print(f"📊 Total documents in ChromaDB: {count}")

if __name__ == "__main__":
    main()

