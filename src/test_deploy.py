import requests
import numpy as np
import os
import json

API_URL = "http://127.0.0.1:8000/predict"

def test_api():
    print("Testing CIKD Deployment API...")
    
    # Try to load a real sample from cache if available
    cache_dir = "data/cache/val"
    if os.path.exists(os.path.join(cache_dir, "text_features.npy")):
        print(f"Loading real data from {cache_dir}")
        text_feats = np.load(os.path.join(cache_dir, "text_features.npy"), mmap_mode='r')[0]
        img_global = np.load(os.path.join(cache_dir, "image_features_global.npy"), mmap_mode='r')[0]
        img_patch = np.load(os.path.join(cache_dir, "image_features_patch.npy"), mmap_mode='r')[0]
        kg_feats = np.load(os.path.join(cache_dir, "kg_features.npy"), mmap_mode='r')[0]
        rel_id = int(np.load(os.path.join(cache_dir, "relation_ids.npy"), mmap_mode='r')[0])
    else:
        print("Real data not found. Using dummy zeros.")
        text_feats = np.zeros(768, dtype=np.float32)
        img_global = np.zeros(512, dtype=np.float32)
        img_patch = np.zeros((49, 512), dtype=np.float32)
        kg_feats = np.zeros(100, dtype=np.float32)
        rel_id = 1

    payload = {
        "text_feats": text_feats.tolist(),
        "img_global": img_global.tolist(),
        "img_patch": img_patch.tolist(),
        "kg_feats": kg_feats.tolist(),
        "relation_id": rel_id
    }
    
    print("Sending POST request to /predict...")
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        print("\n=== API Response ===")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Contradiction Prob: {result['contradiction_prob']:.4f}")
        print(f"Gate Value: {result['gate_value']:.4f}")
        print("Probabilities:")
        for i, p in enumerate(result['probabilities']):
            print(f"  Class {i}: {p:.4f}")
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to API at {API_URL}.")
        print("Make sure the server is running with 'uvicorn src.deploy_api:app'")
    except Exception as e:
        print(f"\nError: {e}")
        if 'response' in locals() and response.text:
            print("Response text:", response.text)

if __name__ == "__main__":
    test_api()
