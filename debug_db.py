#!/usr/bin/env python3
"""
Debug script to check database and configuration status.
Run this to verify that the .env configuration is correct and the database is accessible.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import reset_config, get_config
from src.vector_store.chroma_manager import ChromaDBManager


def main():
    print("=" * 70)
    print("🔍 DATABASE & CONFIGURATION DEBUG")
    print("=" * 70)
    print()
    
    # Force reload config from .env
    reset_config()
    config = get_config()
    
    print("📋 CONFIGURATION (.env)")
    print("-" * 70)
    print(f"  MODEL_NAME: {config.MODEL_NAME}")
    print(f"  VECTOR_DB_PATH: {config.VECTOR_DB_PATH}")
    print(f"  COLLECTION_NAME: {config.COLLECTION_NAME}")
    print(f"  DEVICE: {config.DEVICE}")
    print(f"  MAX_WORKERS: {config.MAX_WORKERS}")
    print(f"  BATCH_SIZE: {config.BATCH_SIZE}")
    print()
    
    # Check database
    print("📦 DATABASE STATUS")
    print("-" * 70)
    
    try:
        manager = ChromaDBManager()
        collections = manager.list_collections()
        
        if not collections:
            print("  ⚠️  No collections found in database!")
            print()
            return
        
        print(f"  Found {len(collections)} collection(s):")
        print()
        
        for col_name in collections:
            stats = manager.get_collection_stats(col_name)
            count = stats.get("count", 0)
            metadata = stats.get("metadata", {})
            model = metadata.get("model_name", "unknown")
            
            # Highlight the collection that matches current model
            is_current = model == config.MODEL_NAME
            marker = "👉 " if is_current else "   "
            
            print(f"{marker}Collection: {col_name}")
            print(f"     Documents: {count}")
            print(f"     Model: {model}")
            if is_current:
                print(f"     ✅ This is the ACTIVE collection for current model")
            print()
        
        # Try loading with current model
        print("🔄 LOADING COLLECTION WITH CURRENT MODEL")
        print("-" * 70)
        collection = manager.get_or_create_collection(model_name=config.MODEL_NAME)
        print(f"  Collection: {collection.name}")
        print(f"  Documents: {collection.count()}")
        print()
        
        if collection.count() == 0:
            print("  ⚠️  WARNING: Collection is empty!")
            print("  This means PostProcessing will show 0 documents.")
            print()
            print("  Possible solutions:")
            print("  1. Check if MODEL_NAME in .env matches the model used during indexing")
            print("  2. Re-run the Load Documents pipeline to index documents")
            print()
        else:
            print(f"  ✅ SUCCESS: {collection.count()} documents ready for PostProcessing")
            print()
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        print()
        import traceback
        traceback.print_exc()
    
    print("=" * 70)
    print("✅ Debug complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
