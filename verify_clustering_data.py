
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.vector_store.chroma_manager import ChromaDBManager

def verify_data():
    try:
        manager = ChromaDBManager()
        # Use auto-detect to find the active collection
        model = manager.auto_detect_model()
        if not model:
            print("No model/collection detected.")
            return

        collection = manager.get_collection_for_model(model)
        print(f"Checking collection: {collection.name}")
        
        # Get a sample
        result = collection.get(limit=10, include=["metadatas"])
        metadatas = result.get("metadatas", [])
        
        clustered_count = 0
        with_keywords = 0
        
        print("\n--- Sample Metadata ---")
        for i, meta in enumerate(metadatas):
            if not meta:
                continue
                
            cid = meta.get("cluster_id")
            if cid is not None:
                clustered_count += 1
                topic = meta.get("cluster_topic", "N/A")
                keywords = meta.get("cluster_keywords", "N/A")
                
                if "cluster_keywords" in meta:
                    with_keywords += 1
                    
                print(f"Doc {i}: Cluster={cid}, Topic='{topic}', Keywords='{keywords}'")
            else:
                print(f"Doc {i}: Not clustered")

        print("\n--- Summary ---")
        print(f"Total checked: {len(metadatas)}")
        print(f"Clustered: {clustered_count}")
        print(f"With Keywords: {with_keywords}")
        
        if with_keywords > 0:
            print("\n✅ SUCCESS: Clustering keywords found in DB.")
        else:
            print("\n⚠️ WARNING: No clustering keywords found. Did you run 'Save to DB' after the update?")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_data()
