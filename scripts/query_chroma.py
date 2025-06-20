#!/usr/bin/env python3
"""
scripts/query_chroma.py - Interactive RAG chat with streaming responses from DeepSeek.

This script connects to the existing ChromaDB, retrieves relevant documents,
and uses the DeepSeek API to generate responses in an interactive chat session.
"""

import os
# Suppress the tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

# --- Rich Print ---
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_storage")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-large-en")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant. Answer the user's questions based on the provided context."
)


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


def print_source_nodes_debug(source_nodes):
    """Prints the source nodes and their full metadata for debugging."""
    if not source_nodes:
        print("\n[No source documents found for this query.]\n")
        return

    print("\n--- Source Documents (Debug) ---")
    for i, source_node in enumerate(source_nodes):
        print(f"📄 Source {i+1} (Score: {source_node.score:.4f})")
        if source_node.node.metadata:
            print("   Metadata:")
            for key, value in source_node.node.metadata.items():
                print(f"     - {key}: {value}")
        
        content = source_node.node.get_content().strip().replace('\n', ' ')
        print(f"   Snippet: '{content[:150]}...'")
        print("-" * 25)
    print("--------------------------------\n")


def print_source_nodes_compressed(source_nodes):
    """Prints the source nodes in a compressed, visual format using Rich."""
    if not source_nodes:
        rprint("\n[bold red][No source documents found for this query.][/bold red]\n")
        return

    rprint("\n[bold]--- Source Documents ---[/bold]")
    for i, source_node in enumerate(source_nodes):
        page_id = source_node.node.metadata.get("page_id", "N/A")
        title = source_node.node.metadata.get("title", "No Title")
        score = source_node.score
        
        pid_suffix = page_id[-3:] if page_id != "N/A" else "N/A"
        
        header = Text()
        header.append(f"{i+1}º ", style="bold white")
        header.append(f"{score:.2f}", style="cyan")
        header.append(" - pID: ", style="white")
        header.append(f"{pid_suffix}", style="yellow")
        header.append(f" - {title}", style="bold magenta")

        content = source_node.node.get_content().strip().replace('\n', ' ')
        snippet = f"'{content[:180]}...'"

        text_content = Text.from_markup(f"[bright_black]{snippet}[/bright_black]")
        text_content.no_wrap = True

        rprint(Panel(
            text_content,
            title=header,
            border_style="dim",
            expand=False
        ))
    rprint("[bold]------------------------[/bold]\n")


def main():
    """Main function to run the interactive RAG chat."""
    parser = argparse.ArgumentParser(description="Interactive RAG chat with DeepSeek.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top similar documents to retrieve.")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model to use (e.g., deepseek-chat, deepseek-coder).")
    parser.add_argument("--persist-dir", default=CHROMA_PERSIST_DIR, help="Directory for ChromaDB persistence.")
    parser.add_argument(
        "--sources-debug",
        action="store_true",
        help="Show full source documents and all metadata for debugging.",
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Show compressed source documents and metadata for the query.",
    )
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        raise ValueError("❌ DEEPSEEK_API_KEY not found in .env file. Please add it.")

    rprint(f"🚀 Starting RAG chat with model: [bold cyan]{args.model}[/bold cyan]")
    rprint(f"💾 Using ChromaDB from: [bold yellow]{args.persist_dir}[/bold yellow]")
    rprint(f"🔍 Retrieving top [bold green]{args.top_k}[/bold green] documents for context.")
    rprint(f"🤖 System Prompt: '[italic]{SYSTEM_PROMPT[:180]}...[/italic]'")

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

    # 5. Create chat engine with memory and system prompt
    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
    
    chat_engine = index.as_chat_engine(
        chat_mode="context",  # type: ignore
        llm=llm,
        similarity_top_k=args.top_k,
        memory=memory,
        system_prompt=SYSTEM_PROMPT,
        streaming=True,
        verbose=False,
    )

    rprint("[grey50]" + "-" * 50 + "[/grey50]")
    rprint("💬 [bold]Ask questions about your Notion documents.[/bold] Type '[bold red]exit[/bold red]' or '[bold red]quit[/bold red]' to end.")

    try:
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    rprint("\n👋 [bold]Goodbye![/bold]")
                    break
                
                if args.sources_debug or args.sources:
                    # To show sources first, we query non-streamed to get nodes, then stream for chat.
                    # This is slightly less efficient as it retrieves twice, but ensures correct order.
                    query_engine = index.as_query_engine(llm=llm, similarity_top_k=args.top_k)
                    response_for_sources = query_engine.query(user_input)
                    
                    if args.sources_debug:
                        print_source_nodes_debug(response_for_sources.source_nodes)
                    elif args.sources:
                        print_source_nodes_compressed(response_for_sources.source_nodes)

                response_stream = chat_engine.stream_chat(user_input)
                print_streaming_response(response_stream)

            except KeyboardInterrupt:
                rprint("\n👋 [bold]Goodbye![/bold]")
                break
            except EOFError:
                rprint("\n👋 [bold]Goodbye![/bold]")
                break
    except Exception as e:
        rprint(f"\n❌ [bold red]An unexpected error occurred:[/bold red] {e}")

if __name__ == "__main__":
    main() 