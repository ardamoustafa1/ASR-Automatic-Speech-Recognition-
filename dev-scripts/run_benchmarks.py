import time
import hashlib
import json
import os
import sys

# Ensure asr_pro is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr_pro.services.conversation_service import analyze_without_save
from asr_pro.db.session import SessionLocal

def run_benchmarks():
    print("🚀 Running God-Tier ASR-Pro Benchmarks...")
    
    # Standard Synthetic Dataset
    dataset = [
        "Uygulamanız dünden beri sürekli çöküyor, paramı geri istiyorum. Aksi takdirde iptal edeceğim.",
        "Kargom gecikti, müşteri hizmetlerini defalarca aradım ama bağlanamadım.",
        "Teşekkür ederim, sorunum çok hızlı çözüldü."
    ]
    
    dataset_str = "|".join(dataset)
    dataset_hash = hashlib.sha256(dataset_str.encode()).hexdigest()[:8]
    
    print(f"Dataset Hash: {dataset_hash}")
    
    results = {
        "dataset_hash": dataset_hash,
        "metrics": {}
    }
    
    # 1. Full Pipeline NLP Benchmark
    db = SessionLocal()
    try:
        start_time = time.time()
        from asr_pro.core.keyword_engine import SegmentInput
        segments = [SegmentInput(start=0, end=0, text=text) for text in dataset]
        analyze_without_save(db, segments)
        nlp_time = time.time() - start_time
        results["metrics"]["full_nlp_pipeline_sec"] = round(nlp_time, 4)
        print(f"✅ Full NLP Pipeline: {nlp_time:.4f}s")
        
        # Total execution
        total_time = nlp_time
        results["metrics"]["total_benchmark_time_sec"] = round(total_time, 4)
        print(f"⏱️ Total Time: {total_time:.4f}s")
    finally:
        db.close()
    
    # Write artifact
    artifact_path = os.path.join(os.getcwd(), "benchmark_results.json")
    with open(artifact_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"📁 Results saved to {artifact_path}")

if __name__ == "__main__":
    run_benchmarks()
