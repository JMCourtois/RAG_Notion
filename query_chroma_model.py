#!/usr/bin/env python3
"""
rag_chat.py - Interactive RAG chat with streaming responses from DeepSeek Chat.

Usage:
  python rag_chat.py [--top-k 5] [--persist-dir <dir>] [--model <model_name>]

Environment variables required in .env:
  DEEPSEEK_API_KEY=<your_deepseek_api_key>

Example:
  python rag_chat.py --top-k 3 --model deepseek-chat
"""
import os
import sys
import argparse
from dotenv import load_dotenv
import time

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings, DEFAULT_TENANT, DEFAULT_DATABASE
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI as BaseLlamaOpenAI
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer

class DeepseekOpenAI(BaseLlamaOpenAI):
    """Subclass to bypass model validation for DeepSeek models."""
    def _get_model_name(self) -> str:
        return "gpt-3.5-turbo"

def print_streaming_response(response_gen):
    """Print streaming response token by token."""
    print("🤖 Assistant: ", end="", flush=True)
    full_response = ""
    
    try:
        # Handle StreamingAgentChatResponse object
        if hasattr(response_gen, 'response_gen'):
            # Check if response_gen is empty or None
            if response_gen.response_gen is None:
                print("[No response generated]")
                return ""
            
            # Access the actual response generator
            for chunk in response_gen.response_gen:
                # Handle direct string chunks
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                    # Add small delay for visible streaming effect
                    time.sleep(0.01)
                # Handle objects with delta attribute
                elif hasattr(chunk, 'delta') and chunk.delta:
                    token = chunk.delta
                    print(token, end="", flush=True)
                    full_response += token
                    time.sleep(0.01)
                # Handle objects with text attribute
                elif hasattr(chunk, 'text') and chunk.text:
                    token = chunk.text
                    print(token, end="", flush=True)
                    full_response += token
                    time.sleep(0.01)
                # Handle objects with content attribute
                elif hasattr(chunk, 'content') and chunk.content:
                    token = chunk.content
                    print(token, end="", flush=True)
                    full_response += token
                    time.sleep(0.01)
        elif hasattr(response_gen, 'response'):
            # Handle non-streaming response
            response_text = response_gen.response
            print(response_text)
            full_response = response_text
        else:
            # Fallback: try to iterate directly
            for chunk in response_gen:
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                    time.sleep(0.01)
                elif hasattr(chunk, 'delta') and chunk.delta:
                    token = chunk.delta
                    print(token, end="", flush=True)
                    full_response += token
                    time.sleep(0.01)
                elif hasattr(chunk, 'text') and chunk.text:
                    token = chunk.text
                    print(token, end="", flush=True)
                    full_response += token
                    time.sleep(0.01)
                elif hasattr(chunk, 'content') and chunk.content:
                    token = chunk.content
                    print(token, end="", flush=True)
                    full_response += token
                    time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n\n⚠️ Response interrupted by user.")
        return full_response
    except Exception as e:
        print(f"\n\n❌ Error during streaming: {e}")
        return full_response
    
    if not full_response:
        print("[No response generated]")
    
    print()  # New line after response
    return full_response

def main():
    parser = argparse.ArgumentParser(description="Interactive RAG chat with streaming responses.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top similar documents to retrieve.")
    parser.add_argument("--persist-dir", default="../chroma_db", help="Directory for ChromaDB persistence.")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model to use.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("❌ DEEPSEEK_API_KEY not found in .env. Please create a .env file with this variable.")

    # Prepare LLM client
    llm = DeepseekOpenAI(
        api_key=api_key,
        api_base="https://api.deepseek.com",
        model=args.model,
        temperature=0.0,
    )

    # Prepare embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")

    # Connect to persistent ChromaDB
    persist_dir = args.persist_dir
    chroma_client = PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    collection = chroma_client.get_or_create_collection("notion_collection")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Check if documents are available
    count = collection.count()
    print(f"📊 Vectors in ChromaDB: {count}")
    if count == 0:
        print("⚠️ No documents indexed. Please run notion_to_chroma.py first.")
        return

    # Hydrate index
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # Create chat engine with memory
    memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
    
    # Try streaming first, fallback to non-streaming
    try:
        chat_engine = index.as_chat_engine(
            llm=llm,
            similarity_top_k=args.top_k,
            memory=memory,
            streaming=True,
            verbose=False
        )
        print("✅ Streaming chat engine created")
    except Exception as e:
        print(f"⚠️ Streaming failed, using non-streaming: {e}")
        chat_engine = index.as_chat_engine(
            llm=llm,
            similarity_top_k=args.top_k,
            memory=memory,
            streaming=False,
            verbose=False
        )
        print("✅ Non-streaming chat engine created")

    print(f"🚀 Interactive RAG Chat started with {args.model}")
    print(f"📚 Using top-{args.top_k} documents for context")
    print(f"💾 ChromaDB: {count} documents available")
    print("💬 Type your questions (Ctrl+C to exit)")
    print("-" * 50)

    try:
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() in ['clear', 'reset']:
                    memory.reset()
                    print("🧹 Chat memory cleared!")
                    continue
                
                if user_input.lower() in ['help', '?']:
                    print("""
📖 Available commands:
- Type your question to chat with the RAG system
- 'clear' or 'reset' - Clear chat memory
- 'exit', 'quit', or 'bye' - Exit the chat
- 'help' or '?' - Show this help message
                    """)
                    continue

                # Handle response (streaming or non-streaming)
                try:
                    response_gen = chat_engine.stream_chat(user_input)
                    print_streaming_response(response_gen)
                except Exception as e:
                    print(f"[DEBUG] Streaming failed: {e}")
                    # Fallback to non-streaming
                    try:
                        response = chat_engine.chat(user_input)
                        print("🤖 Assistant:", response.response)
                    except Exception as e2:
                        print(f"❌ Both streaming and non-streaming failed: {e2}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'exit' to quit.")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

if __name__ == "__main__":
    main() 