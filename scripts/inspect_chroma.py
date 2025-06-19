#!/usr/bin/env python3
"""
scripts/inspect_chroma.py - A utility to inspect the contents of the ChromaDB collection.

This script prints the metadata of all documents (chunks) in the database,
which helps in debugging synchronization and data consistency issues.
"""

import os
import chromadb
from dotenv import load_dotenv
from collections import Counter

# Load environment variables
load_dotenv()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_storage")

def inspect_database():
    """Connects to ChromaDB and prints a summary of its contents."""
    
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"❌ Database directory not found at: {CHROMA_PERSIST_DIR}")
        return

    print(f"🔬 Inspecting ChromaDB at: {CHROMA_PERSIST_DIR}")
    
    try:
        # 1. Connect to the client
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection("notion_collection")

        # 2. Get all documents
        all_docs = collection.get(include=["metadatas"])
        
        total_docs = len(all_docs.get('ids', []))
        print(f"\n📊 Total document chunks found: {total_docs}")

        if total_docs == 0:
            print("✅ The database is empty.")
            return

        # 3. Process metadata
        all_metadatas = all_docs.get('metadatas', [])
        page_ids = [meta.get('page_id', 'N/A') for meta in all_metadatas]
        
        # Count chunks per page_id
        page_id_counts = Counter(page_ids)

        print("\n--- Summary by Page ID ---")
        for page_id, count in page_id_counts.items():
            print(f"- Page ID: {page_id}  |  Chunks: {count}")
        
        print("\n✅ Inspection complete.")

    except Exception as e:
        print(f"\n❌ An error occurred during inspection: {e}")


if __name__ == "__main__":
    inspect_database() 