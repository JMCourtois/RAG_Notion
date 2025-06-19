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