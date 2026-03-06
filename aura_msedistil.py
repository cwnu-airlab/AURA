import os
import sys
import argparse
import json
import torch
from omegaconf import OmegaConf

# Hydra initialization
import hydra
from hydra.core.global_hydra import GlobalHydra

# Import SPLADE modules
# Since we are running from Aura_git/ root, we can just import splade
from splade.train import train
from splade.index import index
from splade.retrieve import retrieve_evaluate


def init_hydra_config(config_name="config_aura_msedistil", overrides=None):
    """
    Programmatically initialize Hydra and return the configuration.
    """
    if overrides is None:
        overrides = []
        
    # Clear any previous hydra instances
    GlobalHydra.instance().clear()
    
    # Initialize hydra in the conf directory
    hydra.initialize(version_base="1.2", config_path="conf")
    
    # Compose the configuration
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    return cfg


def run_training():
    """
    Run the SPLADE training process without the evaluation/validation steps.
    """
    print("\n" + "="*50)
    print("Starting Training (Evaluation Disabled)")
    print("="*50)
    
    cfg = init_hydra_config(config_name="config_aura_msedistil")
    
    # Check current validation settings
    data_cfg = cfg.get("data", {})
    val_loss = data_cfg.get("VALIDATION_SIZE_FOR_LOSS", None)
    val_rank = data_cfg.get("VALIDATION_FULL_RANKING", None)
    
    if val_loss is None and val_rank is None:
        print("Note: Validation settings are null in config. Training will run without evaluation.")
    else:
        print(f"Note: Validation settings detected: loss_size={val_loss}, full_ranking={val_rank}")
        
    try:
        # Call the original train function directly, bypassing the @hydra.main decorator
        train.__wrapped__(cfg)
    except Exception as e:
        print(f"Training stopped or encountered an error: {e}")


def run_evaluation(top_k=10):
    """
    Run the SPLADE indexing and retrieval processes.
    """
    print("\n" + "="*50)
    print("Starting Indexing and Retrieval")
    print("="*50)
    
    cfg = init_hydra_config(config_name="config_aura_msedistil")
    
    try:
        print("1. Running Indexing...")
        # Call the original index function directly
        index.__wrapped__(cfg)
        print("Indexing completed.\n")
        
        print("2. Running Retrieval and Evaluation...")
        # Call the original retrieve_evaluate function directly
        retrieve_evaluate.__wrapped__(cfg)
        print("Retrieval completed.\n")
        
        # 3. View Top-K results
        show_top_k_results(cfg, top_k)
        
    except Exception as e:
        print(f"Evaluation encountered an error: {e}")

def show_top_k_results(cfg, top_k=10):
    """
    Read the output run file and display the top K results.
    """
    print("\n" + "="*50)
    print(f"Extracting Top-{top_k} Retrieval Results")
    print("="*50)
    
    out_dir = cfg.config.out_dir
    run_file_path = os.path.join(out_dir, "run.json")
    
    if not os.path.exists(run_file_path):
        print(f"Error: Run file not found at {run_file_path}")
        print("Please check if retrieval actually generated an output file.")
        
        # Fallback check: Let's see if a .trec file was generated instead
        trec_files = [f for f in os.listdir(out_dir) if f.endswith('.trec')] if os.path.exists(out_dir) else []
        if trec_files:
            print(f"Found .trec files instead: {trec_files}. You might need a trec parser.")
        return
        
    print(f"Reading results from: {run_file_path}\n")
    
    # The output is typically a dictionary mapping query ID to a dict of doc ID -> score
    try:
        with open(run_file_path, 'r') as f:
            results = json.load(f)
            
        # Optional: Read queries to show query text (if available in a specific path)
        # But for now, we just show the raw QIDs and Doc IDs.
            
        print(f"Total queries evaluated: {len(results)}")
        print(f"Showing up to top {top_k} documents for the first 3 queries as an example:\n")
        
        query_count = 0
        for qid, doc_scores in results.items():
            if query_count >= 3:
                break
                
            print(f"Query ID: {qid}")
            
            # Sort documents by score (descending)
            sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
            
            for i, (doc_id, score) in enumerate(sorted_docs[:top_k]):
                print(f"  Rank {i+1:2d} | Doc ID: {doc_id:<15} | Score: {score:.4f}")
            
            print("-" * 40)
            query_count += 1
            
    except Exception as e:
        print(f"Error formatting results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Aura/SPLADE Runner")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "all", "results"], 
                        default="all", help="Mode to run: train, eval, all, or results (view results only)")
    parser.add_argument("--top_k", type=int, default=10, 
                        help="Number of top documents to show in results view")
    
    args = parser.parse_args()
    
    # Important: add the Aura directory to path so splade modules can be imported correctly
    # without breaking internal relative imports
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    if args.mode in ["train", "all"]:
        run_training()
        
    if args.mode in ["eval", "all"]:
        run_evaluation(top_k=args.top_k)
        
    if args.mode == "results":
        # Just initialize hydra to get the config and show the results
        cfg = init_hydra_config(config_name="config_aura_msedistil")
        show_top_k_results(cfg, top_k=args.top_k)
