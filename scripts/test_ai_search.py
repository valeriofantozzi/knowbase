import sys
from pathlib import Path
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_search.graph import build_graph
from src.utils.config import Config

def test_ai_search():
    print("Testing AI Search Pipeline...")
    
    config = Config()
    if not config.OPENAI_API_KEY:
        print("Skipping test: OPENAI_API_KEY not found in environment.")
        return

    try:
        app = build_graph()
        print("Graph built successfully.")
        
        question = "What are the benefits of orchids?"
        print(f"Invoking with question: '{question}'")
        
        initial_state = {
            "messages": [],
            "question": question
        }
        
        response = app.invoke(initial_state)
        
        print("\n--- Result ---")
        print(f"Rewritten Question: {response.get('question')}")
        print(f"Retrieved Documents: {len(response.get('documents', []))}")
        print(f"Answer: {response.get('generation')}")
        print("----------------")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ai_search()
