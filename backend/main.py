import os
import yaml
import torch
import sys
import argparse
from transformers import AutoTokenizer
from src.dataset import load_amazon_split, load_vsfc, load_tweeteval, make_dataloader, load_multi_domain_amazon
from src.model import BaseModel, DANNModel
from src.train import train_model, train_dann, compute_class_weights
from src.evaluate import evaluate_model, print_summary_table, save_research_results
from src.utils import set_seed, print_banner, save_model, load_model_weights

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=0, help="Scenario number to run (1-6). 0 for ALL.")
    args = parser.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(os.path.join(current_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    
    max_tr = cfg["scenarios"]["max_samples_train"]
    max_te = cfg["scenarios"]["max_samples_test"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    results = {}
    
    # Checkpoint paths
    cp_dir = os.path.join(current_dir, "checkpoints")
    path_s1 = os.path.join(cp_dir, "model_s1.pt")
    path_s3 = os.path.join(cp_dir, "model_s3.pt")

    # --- S1: Baseline ---
    if args.s in [0, 1]:
        print_banner("Scenario 1: Baseline Training")
        tr_texts, tr_labels, tr_d_ids = load_amazon_split("english", "books", "train", max_samples=max_tr)
        train_loader = make_dataloader(tr_texts, tr_labels, tr_d_ids, tokenizer, batch_size=cfg["training"]["batch_size"], shuffle=True)
        model = BaseModel(model_name=cfg["model"]["name"], num_labels=cfg["model"]["num_labels"])
        model = train_model(model, tokenizer, train_loader, num_epochs=cfg["training"]["epochs"], device=device)
        save_model(model, path_s1)
        
        te_texts, te_labels, te_d_ids = load_amazon_split("english", "books", "test", max_samples=max_te)
        test_loader = make_dataloader(te_texts, te_labels, te_d_ids, tokenizer, batch_size=cfg["training"]["batch_size"])
        results["1. Baseline"] = evaluate_model(model, test_loader, device=device, scenario_name="Baseline")

    # --- S2: Cross-domain (Requires S1) ---
    if args.s in [0, 2]:
        print_banner("Scenario 2: Cross-domain Evaluation")
        model = BaseModel(model_name=cfg["model"]["name"], num_labels=cfg["model"]["num_labels"])
        model = load_model_weights(model, path_s1, device=device)
        te_texts, te_labels, te_d_ids = load_amazon_split("english", "electronics", "test", max_samples=max_te)
        test_loader = make_dataloader(te_texts, te_labels, te_d_ids, tokenizer, batch_size=cfg["training"]["batch_size"])
        results["2. Cross-domain"] = evaluate_model(model, test_loader, device=device, scenario_name="Cross-domain")

    # --- S3: Multi-domain MDL ---
    if args.s in [0, 3]:
        print_banner("Scenario 3: Multi-domain MDL Training")
        tr_texts, tr_labels, tr_d_ids = load_multi_domain_amazon(["books", "electronics", "apparel"], max_samples=max_tr)
        train_loader = make_dataloader(tr_texts, tr_labels, tr_d_ids, tokenizer, batch_size=cfg["training"]["batch_size"], shuffle=True)
        model = BaseModel(model_name=cfg["model"]["name"], num_labels=cfg["model"]["num_labels"])
        model = train_model(model, tokenizer, train_loader, num_epochs=cfg["training"]["epochs"], device=device)
        save_model(model, path_s3)
        
        te_texts, te_labels, te_d_ids = load_amazon_split("english", "electronics", "test", max_samples=max_te)
        test_loader = make_dataloader(te_texts, te_labels, te_d_ids, tokenizer, batch_size=cfg["training"]["batch_size"])
        results["3. Multi-domain (MDL)"] = evaluate_model(model, test_loader, device=device, scenario_name="MDL")

    # --- S4: DANN Adaptation ---
    if args.s in [0, 4]:
        print_banner("Scenario 4: DANN Adaptation")
        tr_texts, tr_labels, tr_d_ids = load_multi_domain_amazon(["books", "electronics", "apparel"], max_samples=max_tr)
        source_loader = make_dataloader(tr_texts, tr_labels, tr_d_ids, tokenizer, batch_size=cfg["training"]["batch_size"]//2, shuffle=True)
        tu_texts, tu_labels, tu_d_ids = load_tweeteval("train", max_samples=max_tr, unlabeled=True)
        target_loader = make_dataloader(tu_texts, tu_labels, tu_d_ids, tokenizer, batch_size=cfg["training"]["batch_size"]//2, shuffle=True)
        
        model = DANNModel(model_name=cfg["model"]["name"], num_labels=cfg["model"]["num_labels"], num_domains=cfg["model"]["num_domains"])
        model = train_dann(model, tokenizer, source_loader, target_loader, num_epochs=cfg["training"]["epochs"], device=device)
        save_model(model, os.path.join(cp_dir, "model_s4.pt"))
        
        te_texts, te_labels, te_d_ids = load_tweeteval("test", max_samples=max_te)
        test_loader = make_dataloader(te_texts, te_labels, te_d_ids, tokenizer, batch_size=cfg["training"]["batch_size"])
        results["4. DANN Adaptation"] = evaluate_model(model, test_loader, device=device, scenario_name="DANN Adaptation")

    # --- S5: Cross-lingual (Requires S3) ---
    if args.s in [0, 5]:
        print_banner("Scenario 5: Cross-lingual Evaluation")
        model = BaseModel(model_name=cfg["model"]["name"], num_labels=cfg["model"]["num_labels"])
        model = load_model_weights(model, path_s3, device=device)
        te_texts, te_labels, te_d_ids = load_vsfc("test", max_samples=max_te)
        test_loader = make_dataloader(te_texts, te_labels, te_d_ids, tokenizer, batch_size=cfg["training"]["batch_size"])
        results["5. Cross-lingual"] = evaluate_model(model, test_loader, device=device, scenario_name="Cross-lingual")

    # --- S6: Multilingual ---
    if args.s in [0, 6]:
        print_banner("Scenario 6: Multilingual Training")
        en_texts, en_labels, en_d_ids = load_multi_domain_amazon(["books", "electronics", "apparel"], max_samples=max_tr//2)
        vi_texts, vi_labels, vi_d_ids = load_vsfc("train", max_samples=max_tr//2)
        train_loader = make_dataloader(en_texts + vi_texts, en_labels + vi_labels, en_d_ids + vi_d_ids, tokenizer, batch_size=cfg["training"]["batch_size"], shuffle=True)
        model = BaseModel(model_name=cfg["model"]["name"], num_labels=cfg["model"]["num_labels"])
        model = train_model(model, tokenizer, train_loader, num_epochs=cfg["training"]["epochs"], device=device)
        
        te_texts, te_labels, te_d_ids = load_vsfc("test", max_samples=max_te)
        test_loader = make_dataloader(te_texts, te_labels, te_d_ids, tokenizer, batch_size=cfg["training"]["batch_size"])
        results["6. Multilingual"] = evaluate_model(model, test_loader, device=device, scenario_name="Multilingual")

    if results:
        save_research_results(results, os.path.join(current_dir, "results", f"results_s{args.s}.json"))

if __name__ == "__main__":
    main()
