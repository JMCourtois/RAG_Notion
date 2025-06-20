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
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.node_parser import SentenceSplitter

# Load environment variables
load_dotenv()
DEFAULT_NOTION_TOKEN = os.getenv("NOTION_INTEGRATION_TOKEN")
DEFAULT_NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_storage")
DEFAULT_HISTORY_PATH = os.getenv("CHROMA_HISTORY_PATH", "./data/indexed_pages.json")

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Ingest Notion into ChromaDB for RAG.")
parser.add_argument("--notion-token", default=DEFAULT_NOTION_TOKEN, help="Notion integration token.")
parser.add_argument("--notion-page-id", default=DEFAULT_NOTION_PAGE_ID, help="Root Notion page ID.")
parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="ChromaDB persistence directory.")
parser.add_argument("--history-path", default=DEFAULT_HISTORY_PATH, help="Indexing history JSON path.")
parser.add_argument("--force-reindex", action="store_true", help="Force reindexing even if no changes detected.")
parser.add_argument("--reset-chroma", action="store_true", help="Delete existing ChromaDB database and history before running.")
args = parser.parse_args()

# --- Global Variables & Constants ---
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
console = Console()

# --- Helper Functions ---

def load_history(history_path):
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
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

def _get_block_text(block) -> str:
    """Extracts plain text from a Notion block."""
    block_type = block.get("type")
    if not block_type:
        return ""
    
    content = block.get(block_type, {})
    
    # Check for rich_text which is the most common text container
    if "rich_text" in content:
        return "".join(text.get("plain_text", "") for text in content["rich_text"])
        
    # Fallback for other potential text representations (less common)
    if "text" in content and isinstance(content["text"], list):
        return "".join(t.get("plain_text", "") for t in content["text"])

    return ""


def _get_all_blocks_text(page_id: str, progress, task_id, page_title: str) -> str:
    """Paginates through all blocks of a page and concatenates their text, updating progress."""
    full_text = []
    block_count = 0
    next_cursor = None
    while True:
        response = notion.blocks.children.list(block_id=page_id, start_cursor=next_cursor, page_size=100)
        blocks = response.get("results", [])  # type: ignore
        block_count += len(blocks)
        
        # Update progress bar description with block count
        progress.update(task_id, description=f"Reading '{page_title}' ({block_count} blocks)")

        for block in blocks:
            full_text.append(_get_block_text(block))
        
        next_cursor = response.get("next_cursor")  # type: ignore
        if not next_cursor:
            break
            
    return "\n".join(filter(None, full_text))


def _discover_child_pages_recursive(block_id: str, discovered_pages: set):
    """Recursively finds all child page IDs, providing real-time feedback with titles."""
    try:
        next_cursor = None
        while True:
            response = notion.blocks.children.list(block_id=block_id, start_cursor=next_cursor, page_size=100)
            blocks = response.get("results", []) # type: ignore
            for block in blocks:
                if block.get("type") == "child_page":
                    child_page_id = block.get("id") # type: ignore
                    if child_page_id not in discovered_pages:
                        discovered_pages.add(child_page_id)
                        try:
                            # Retrieve page details to get the title for real-time feedback
                            page_info = notion.pages.retrieve(page_id=child_page_id)
                            title_list = page_info.get("properties", {}).get("title", {}).get("title", []) # type: ignore
                            title = title_list[0].get("plain_text") if title_list else "Untitled" # type: ignore
                            console.print(f"  [dim] -> Discovered page:[/dim] [cyan]{title}[/cyan]")
                        except Exception:
                            # Fallback to printing ID if title retrieval fails
                            console.print(f"  [dim] -> Discovered page ID:[/dim] [cyan]{child_page_id}[/cyan]")
                        
                        # Recurse into the newly found page
                        _discover_child_pages_recursive(child_page_id, discovered_pages)
                elif block.get("has_children"): # type: ignore
                    _discover_child_pages_recursive(block.get("id"), discovered_pages) # type: ignore
            
            next_cursor = response.get("next_cursor") # type: ignore
            if not next_cursor:
                break
    except Exception as e:
        console.print(f"  [red]⚠️ Error exploring children of block {block_id}: {e}[/red]")

def read_and_enrich_pages(page_ids: list) -> list:
    """
    Reads content and metadata for a list of Notion page IDs, showing progress.
    This replaces both NotionPageReader and the separate enrichment step.
    """
    all_docs = []
    with Progress(
        SpinnerColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("[cyan]{task.description}[/cyan] [bold]({task.completed} of {task.total})[/bold]"),
    ) as progress:
        task = progress.add_task("[green]Reading Pages...", total=len(page_ids))
        for page_id in page_ids:
            try:
                # 1. Get Page Metadata
                page_info = notion.pages.retrieve(page_id=page_id)
                title_list = page_info.get("properties", {}).get("title", {}).get("title", []) # type: ignore
                title = title_list[0].get("plain_text") if title_list else "Untitled" # type: ignore
                
                progress.update(task, description=f"Reading '{title}'")
                
                # 2. Get Page Content (passing progress and task details)
                content = _get_all_blocks_text(page_id, progress, task, title)
                
                # 3. Create LlamaIndex Document
                doc_metadata = {
                    "page_id": page_id,
                    "title": title,
                    "last_edited_time": page_info.get("last_edited_time"), # type: ignore
                    "url": page_info.get("url") # type: ignore
                }
                
                # Filter out any None values from metadata
                doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None}
                
                doc = Document(text=content, metadata=doc_metadata)
                all_docs.append(doc)
                
            except Exception as e:
                console.print(f"\n[red]⚠️ Failed to process page {page_id}: {e}[/red]")
            
            progress.update(task, advance=1)
            
    return all_docs

def main():
    start_time = time.monotonic()
    
    console.print(Panel("[bold green]🚀 Notion to ChromaDB Indexer 🚀[/bold green]", expand=False))

    if args.reset_chroma:
        console.print("🔥 [bold yellow]--reset-chroma[/bold yellow] flag detected. Deleting database and history.")
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
            console.print(f"✅ Deleted ChromaDB directory: [cyan]{PERSIST_DIR}[/cyan]")
        data_dir = os.path.dirname(HISTORY_PATH)
        if os.path.exists(data_dir):
             shutil.rmtree(data_dir)
             console.print(f"✅ Deleted history directory: [cyan]{data_dir}[/cyan]")
        console.print("[green]✅ Reset complete. Proceeding with fresh indexing...[/green]")

    # --- 1. Discover Pages ---
    with console.status("[bold green]🔍 Discovering pages and subpages...") as status:
        all_page_ids = {NOTION_PAGE_ID}
        _discover_child_pages_recursive(NOTION_PAGE_ID, all_page_ids)
        page_ids_list = list(all_page_ids)
    console.print(f"📄 [bold]Found a total of {len(page_ids_list)} pages.[/bold]")

    history = load_history(HISTORY_PATH)
    
    # --- 2. Sync Deleted Pages ---
    if not FORCE_REINDEX:
        indexed_page_ids = set(history.keys())
        discovered_page_ids = set(page_ids_list)
        deleted_page_ids = indexed_page_ids - discovered_page_ids
        
        if deleted_page_ids:
            console.print(f"🗑️ [bold yellow]Found {len(deleted_page_ids)} page(s) deleted in Notion. Purging from ChromaDB...[/bold yellow]")
            chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
            chroma_collection = chroma_client.get_or_create_collection("notion_collection")
            chroma_collection.delete(where={"page_id": {"$in": list(deleted_page_ids)}})
            
            # Update history by removing deleted pages
            for page_id in deleted_page_ids:
                del history[page_id]
            save_history(history, HISTORY_PATH)
            console.print("✅ [green]Purge complete.[/green]")

    # --- 3. Read & Enrich Documents ---
    console.print("📖 [bold]Reading content and metadata for all pages...[/bold]")
    all_docs = read_and_enrich_pages(page_ids_list)
    console.print(f"✅ Loaded and enriched {len(all_docs)} documents.")

    # --- 4. Filter Documents ---
    is_db_empty = not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR)
    if is_db_empty: console.print("⚠️ [yellow]ChromaDB is empty. Indexing all documents.[/yellow]")

    if FORCE_REINDEX or is_db_empty:
        docs_to_index = all_docs
    else:
        with console.status("[bold green]🔎 Checking for modifications...[/bold]"):
            docs_to_index = [doc for doc in all_docs if should_reindex(doc, history)]
    
    if not docs_to_index and not deleted_page_ids:
        console.print("\n[green]✅ All documents are up to date. No new indexing required.[/green]")
        return
    elif not docs_to_index and deleted_page_ids:
         console.print("\n[green]✅ Purge complete. No new documents to index.[/green]")
         return

    console.print(f"🆕 [bold]Found {len(docs_to_index)} new or modified documents to index.[/bold]")

    # --- 5. Initialize Services ---
    console.print("💾 [bold]Initializing ChromaDB and loading embedding model...[/bold]")
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection("notion_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")
    
    # --- FIX: Set embed_model in global settings to avoid fallback to OpenAI ---
    Settings.embed_model = embed_model

    # --- 6. Index Documents ---
    if not is_db_empty and not FORCE_REINDEX:
        page_ids_to_update = {doc.metadata['page_id'] for doc in docs_to_index}
        console.print(f"🔄 [yellow]Deleting old versions for {len(page_ids_to_update)} updated document(s)...[/yellow]")
        chroma_collection.delete(where={"page_id": {"$in": list(page_ids_to_update)}})

    console.print(f"📚 [bold]Indexing {len(docs_to_index)} documents into ChromaDB...[/bold] (this is the heavy part!)")

    # Manual, stylized node parsing and embedding
    with Progress(SpinnerColumn(), BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TextColumn("{task.description}")) as progress:
        # Step 5.1: Parse documents into nodes with optimal, industry-standard settings
        console.print("   [dim]Chunking documents with size=512, overlap=50...[/dim]")
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        parsing_task = progress.add_task("[green]Parsing documents...", total=len(docs_to_index))
        nodes = []
        for doc in docs_to_index:
            nodes.extend(parser.get_nodes_from_documents([doc]))
            progress.update(parsing_task, advance=1)
        
        # Step 5.2: Generate embeddings for nodes
        embedding_task = progress.add_task("[magenta]Generating embeddings...", total=len(nodes))
        for node in nodes:
            node.embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
            progress.update(embedding_task, advance=1)

    # Step 5.3: Add nodes to the index
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    
    # --- 7. Update History ---
    console.print("📝 [bold]Updating history log...[/bold]")
    for doc in docs_to_index:
        pid = doc.metadata.get("page_id")
        last_edit = doc.metadata.get("last_edited_time")
        if pid and last_edit: history[pid] = last_edit
    save_history(history, HISTORY_PATH)

    # --- 8. Final Summary ---
    end_time = time.monotonic()
    total_time = end_time - start_time
    
    summary_table = Table(title="🎉 Indexing Complete! 🎉", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="dim", width=25)
    summary_table.add_column("Value", style="bold")
    
    summary_table.add_row("Pages Found", str(len(page_ids_list)))
    summary_table.add_row("Documents Indexed", str(len(docs_to_index)))
    summary_table.add_row("Total Docs in DB", str(chroma_collection.count()))
    summary_table.add_row("Total Time", f"{total_time:.2f} seconds")
    
    console.print(summary_table)

if __name__ == "__main__":
    main()