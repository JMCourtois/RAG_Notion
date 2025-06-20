#!/usr/bin/env python3
"""
scripts/inspect_chroma.py - A utility to inspect the contents of the ChromaDB collection.

This script prints a summary of documents grouped by their source Notion page
in a clean, readable table format.
"""

import os
import chromadb
from dotenv import load_dotenv
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Load environment variables
load_dotenv()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_storage")
console = Console()

def inspect_database():
    """Connects to ChromaDB and prints a summary of its contents in a table."""
    
    if not os.path.exists(CHROMA_PERSIST_DIR):
        console.print(Panel(f"❌ [bold red]Database directory not found[/bold red]\n[dim]{CHROMA_PERSIST_DIR}[/dim]", 
                            title="Error", expand=False, border_style="red"))
        return

    console.print(Panel(f"🔬 Inspecting ChromaDB at [cyan]{CHROMA_PERSIST_DIR}[/cyan]", 
                        title="[bold green]ChromaDB Inspector[/bold green]", expand=False))
    
    try:
        # 1. Connect to the client
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection("notion_collection")

        # 2. Get all documents
        all_docs = collection.get(include=["metadatas"])
        
        total_docs = len(all_docs.get('ids', []))
        
        if total_docs == 0:
            console.print("\n[bold green]✅ The database is empty.[/bold green]")
            return

        # 3. Group chunks by page_id and aggregate metadata
        pages_summary = defaultdict(lambda: {'count': 0, 'title': 'N/A'})
        all_metadatas = all_docs.get('metadatas', []) or []
        
        for meta in all_metadatas:
            page_id = str(meta.get('page_id', 'Unknown_ID'))
            # Ensure count is treated as an integer before incrementing
            current_count = int(pages_summary[page_id].get('count', 0))
            pages_summary[page_id]['count'] = current_count + 1
            
            if meta.get('title'):
                pages_summary[page_id]['title'] = str(meta.get('title'))

        # 4. Create and display the table
        table = Table(title=f"📊 Summary: {total_docs} Chunks Across {len(pages_summary)} Pages",
                      show_header=True, header_style="bold magenta")
        table.add_column("PageID", style="dim", width=6)
        table.add_column("Page Title", style="cyan", no_wrap=False)
        table.add_column("Chunk Count", justify="right", style="green")

        # Sort pages by title for consistent order
        sorted_pages = sorted(pages_summary.items(), key=lambda item: item[1]['title'])

        for page_id, data in sorted_pages:
            # Show last 3 chars of ID for quick reference
            short_id = f"...{page_id[-3:]}" if len(page_id) > 3 else page_id
            table.add_row(short_id, str(data['title']), str(data['count']))
        
        console.print(table)
        console.print("\n[bold green]✅ Inspection complete.[/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]❌ An error occurred during inspection:[/bold red] {e}")


if __name__ == "__main__":
    inspect_database() 