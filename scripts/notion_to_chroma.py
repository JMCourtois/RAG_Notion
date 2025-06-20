#!/usr/bin/env python3
"""
scripts/notion_to_chroma.py - Ingest Notion pages (and sub-pages) into ChromaDB for RAG pipelines.
Configurable via .env variables and/or command-line arguments.
"""
import os
import json
import argparse
import shutil
import time

from dotenv import load_dotenv
from notion_client import Client as NotionClient
import chromadb

# --- Rich UI Imports ---
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

# --- LlamaIndex Imports ---
from llama_index.readers.notion import NotionPageReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings, DEFAULT_TENANT, DEFAULT_DATABASE
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter

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
console = Console() # Rich console for better UI

def load_history(history_path):
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        console.print(f"⚠️ The history file {history_path} is empty or corrupted. It will be reset.")
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
    Recursively finds all child page IDs, providing real-time feedback with titles.
    """
    try:
        next_cursor = None
        while True:
            response = notion.blocks.children.list(block_id=block_id, start_cursor=next_cursor, page_size=100)
            blocks = response.get("results", []) # type: ignore
            for block in blocks:
                if block.get("type") == "child_page":
                    child_page_id = block.get("id")
                    if child_page_id not in discovered_pages:
                        discovered_pages.add(child_page_id)
                        try:
                            # Retrieve page details for real-time feedback
                            page_info = notion.pages.retrieve(page_id=child_page_id)
                            title_list = page_info.get("properties", {}).get("title", {}).get("title", []) # type: ignore
                            title = title_list[0].get("plain_text") if title_list else "Untitled"
                            console.print(f"  [dim] -> Discovered page:[/dim] [cyan]{title}[/cyan]")
                        except Exception:
                            console.print(f"  [dim] -> Discovered page ID:[/dim] [cyan]{child_page_id}[/cyan]")
                        
                        _discover_child_pages_recursive(child_page_id, discovered_pages)
                elif block.get("has_children"):
                    _discover_child_pages_recursive(block.get("id"), discovered_pages)
            
            next_cursor = response.get("next_cursor") # type: ignore
            if not next_cursor:
                break
    except Exception as e:
        console.print(f"  [red]⚠️ Error exploring children of block {block_id}: {e}[/red]")

def sync_deleted_pages(chroma_collection, history: dict, notion_page_ids: set):
    """
    Removes documents from ChromaDB and history if their source page no longer exists in Notion.
    This function performs a full audit of the database.
    """
    console.print("🔄 [bold]Syncing deleted pages...[/bold]")
    
    # 1. Get ALL documents from ChromaDB to perform a full audit.
    try:
        all_chroma_docs = chroma_collection.get(include=["metadatas"])
        all_doc_ids = all_chroma_docs.get('ids', [])
        all_metadatas = all_chroma_docs.get('metadatas', [])
    except Exception as e:
        console.print(f"⚠️ Could not get documents from ChromaDB to sync deletes: {e}")
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
        console.print(f"🗑️ Found {len(doc_ids_to_delete)} orphaned document chunks to delete...")
        try:
            chroma_collection.delete(ids=doc_ids_to_delete)
            console.print("✅ Successfully deleted orphaned chunks from ChromaDB.")
        except Exception as e:
            console.print(f"❌ Error deleting chunks from ChromaDB: {e}")
    else:
        console.print("✅ No orphaned document chunks found in ChromaDB.")

    # 4. Clean up the history file based on what's no longer in Notion
    history_page_ids_to_delete = set(history.keys()) - notion_page_ids
    if history_page_ids_to_delete:
        console.print(f"🗑️ Found {len(history_page_ids_to_delete)} page(s) to remove from history...")
        for page_id in history_page_ids_to_delete:
            del history[page_id]
        console.print("✅ Successfully removed pages from history.")
    else:
        console.print("✅ History file is already up to date.")

def enrich_document_metadata(doc, notion_client):
    pid = doc.metadata.get("page_id")
    if pid is None:
        return doc
    try:
        page_info = notion_client.pages.retrieve(page_id=pid)
    except Exception as e:
        console.print(f"⚠️ Error retrieving metadata for {pid}: {e}")
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
    start_time = time.monotonic()
    console.print(Panel("[bold green]🚀 Notion to ChromaDB Indexer 🚀[/bold green]", expand=False))
    
    if args.reset_chroma:
        console.print("🔥 --reset-chroma flag detected. Deleting database and history.")
        if os.path.exists(PERSIST_DIR):
            try:
                shutil.rmtree(PERSIST_DIR)
                console.print(f"✅ Deleted ChromaDB directory: {PERSIST_DIR}")
            except OSError as e:
                console.print(f"❌ Error deleting directory {PERSIST_DIR}: {e}")
        else:
            console.print(f"🤷 Directory {PERSIST_DIR} not found, nothing to delete.")
        
        history_dir = os.path.dirname(HISTORY_PATH)
        if os.path.exists(history_dir):
            try:
                shutil.rmtree(history_dir)
                console.print(f"✅ Deleted history directory: {history_dir}")
            except OSError as e:
                console.print(f"❌ Error deleting directory {history_dir}: {e}")
        else:
            console.print(f"🤷 Directory {history_dir} not found, nothing to delete.")

        console.print("✅ Reset complete. Proceeding with fresh indexing...")

    console.print(f"🔗 Notion page ID: {NOTION_PAGE_ID}")
    console.print(f"💾 ChromaDB dir: {PERSIST_DIR}")
    console.print(f"📝 History: {HISTORY_PATH}")
    if FORCE_REINDEX:
        console.print("🔄 Forced reindexing enabled")

    history = load_history(HISTORY_PATH)

    # 1) Discover all pages and subpages
    with console.status("[bold green]🔍 Discovering pages and subpages...") as status:
        all_page_ids = {NOTION_PAGE_ID}
        _discover_child_pages_recursive(NOTION_PAGE_ID, all_page_ids)
    console.print(f"📄 [bold]Found {len(all_page_ids)} pages (including subpages)[/bold]")

    # Connect to ChromaDB and load history before syncing
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection("notion_collection")

    # 2) Sync and remove deleted pages before doing anything else
    sync_deleted_pages(chroma_collection, history, all_page_ids)
    
    # 3) Read Notion documents for the remaining pages
    reader = NotionPageReader(integration_token=NOTION_TOKEN)
    console.print("📖 [bold]Reading Notion pages...[/bold]")
    docs = reader.load_data(page_ids=list(all_page_ids))
    console.print(f"✅ Loaded {len(docs)} Notion documents.")

    # 4) Enrich metadata with a progress bar
    console.print("🧠 [bold]Processing and enriching documents...[/bold]")
    enriched_docs = []
    with Progress(
        SpinnerColumn(), 
        BarColumn(), 
        "[progress.percentage]{task.percentage:>3.0f}%", 
        TextColumn("[cyan]{task.description}[/cyan] [bold]({task.completed} of {task.total})[/bold]"),
        transient=True
    ) as progress:
        task = progress.add_task("[green]Processing pages...", total=len(docs))
        for doc in docs:
            enriched_doc = enrich_document_metadata(doc, notion)
            title = enriched_doc.metadata.get("title", f"Page ID: {enriched_doc.metadata.get('page_id', 'N/A')}")
            progress.update(task, advance=1, description=f"Processing '{title}'")
            enriched_docs.append(enriched_doc)
    
    console.print(f"✅ Processed and enriched {len(enriched_docs)} documents.")

    # 5) Configure LlamaIndex Settings (Chunking and Embedding)
    console.print("⚙️ [bold]Configuring LlamaIndex Settings...[/bold]")
    
    # Using the default values so they are visible and can be easily changed.
    # Standard chunk size, good for general context.
    chunk_size = 1024 
    # Overlap to avoid losing context between chunks.
    chunk_overlap = 200 

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    console.print(f"  [dim]‣ Chunk Size:[/dim] [cyan]{chunk_size} tokens[/cyan]")
    console.print(f"  [dim]‣ Chunk Overlap:[/dim] [cyan]{chunk_overlap} tokens[/cyan]")
    console.print(f"  [dim]‣ Embedding Model:[/dim] [cyan]BAAI/bge-large-en[/cyan]")

    # 6) Persistent Chroma client
    console.print(f"💾 Initializing ChromaDB at {PERSIST_DIR}...")
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
        console.print("⚠️ ChromaDB is empty. Forcing a re-index of all documents.")

    # 7) Filter docs to index
    if FORCE_REINDEX or is_db_empty:
        docs_to_index = enriched_docs
        if FORCE_REINDEX:
            console.print(f"🔄 Reindexing {len(docs_to_index)} documents (forced by flag)")
        else:
            console.print(f"🔄 Reindexing {len(docs_to_index)} documents (database was empty)")
    else:
        docs_to_index = [doc for doc in enriched_docs if should_reindex(doc, history)]
        console.print(f"🆕 Documents to index based on last edit time: {len(docs_to_index)}")

    # 8) Cleanly update the index by first deleting old chunks
    if not is_db_empty and docs_to_index:
        page_ids_to_update = {doc.metadata['page_id'] for doc in docs_to_index}
        console.print(f"🔄 Preparing to update {len(page_ids_to_update)} page(s) by deleting their old chunks first...")
        # Ensure the list of page IDs are strings for the filter
        delete_filter = {"page_id": {"$in": [str(pid) for pid in page_ids_to_update]}}
        try:
            chroma_collection.delete(where=delete_filter) # type: ignore
            console.print("✅ Successfully deleted old chunks for pages to be updated.")
        except Exception as e:
            console.print(f"❌ Error deleting old chunks during update: {e}")

    # 9) Index or ensure the index
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    console.print("📚 Creating/updating index in ChromaDB...")
    if docs_to_index:
        index = VectorStoreIndex.from_documents(
            docs_to_index,
            storage_context=storage_context,
            # embed_model and node_parser are sourced from global Settings
        )
        console.print(f"✅ Indexed {len(docs_to_index)} new/modified documents.")
    else:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            # embed_model is sourced from global Settings
        )
        console.print("✅ Existing documents available in the index.")

    # 10) Persist changes
    index.storage_context.persist()
    console.print("✅ Index saved to ChromaDB.")

    # 11) Update history
    for doc in docs_to_index:
        pid = doc.metadata.get("page_id")
        last_edit = doc.metadata.get("last_edited_time")
        if pid and last_edit:
            history[pid] = last_edit
    if docs_to_index:
        save_history(history, HISTORY_PATH)
        console.print(f"✅ History updated ({len(docs_to_index)} pages indexed).")
    else:
        console.print("📝 No new pages to update in history.")

    # 12) Check documents in ChromaDB
    count = chroma_collection.count()
    console.print(f"📊 Total documents in ChromaDB: {count}")

    # 12) Final Summary Table
    end_time = time.monotonic()
    total_time = end_time - start_time
    
    summary_table = Table(title="Indexing Complete! 🎉", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="dim", width=25)
    summary_table.add_column("Value", style="bold")
    
    summary_table.add_row("Pages Found", str(len(all_page_ids)))
    summary_table.add_row("Docs Indexed/Updated", str(len(docs_to_index)))
    summary_table.add_row("Total Docs in DB", str(chroma_collection.count()))
    summary_table.add_row("Total Time", f"{total_time:.2f} seconds")
    
    console.print(summary_table)

if __name__ == "__main__":
    main()