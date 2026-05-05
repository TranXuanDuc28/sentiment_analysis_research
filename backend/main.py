import argparse
import yaml
import os
import sys
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from src.dataset import load_amazon_split, load_vsfc, load_yelp, load_imdb, make_dataloader
from src.model import BaseModel, DANNModel
from src.train import train_model, train_dann, compute_class_weights
from src.evaluate import evaluate_model
from src.report_generator import generate_aggregate_report
from src.utils import print_banner, save_results, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=str, default="0", help="Scenario to run (0=all, 1a, 1b, 2, 3, 4, 5, 6, 7)")
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    MAX_TEST = max(config["scenarios"]["max_samples_test"], 1000)
    MAX_TRAIN = config["scenarios"]["max_samples_train"]
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = int(config["training"]["epochs"])
    LR = float(config["training"]["learning_rate"])
    
    global_results = {}
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    # --- S1a: Monolingual Source Baseline (IMDb) ---
    if args.s in ["0", "1a"]:
        print_banner("Scenario 1a: Monolingual Source Baseline (IMDb)")
        t_all, l_all, d_all = load_imdb("train", max_samples=MAX_TRAIN)
        t_train, t_val, l_train, l_val, d_train, d_val = train_test_split(t_all, l_all, d_all, test_size=0.1, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, tokenizer, batch_size=BATCH_SIZE)
        
        model = BaseModel(config["model"]["name"])
        weights = compute_class_weights(l_train)
        model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
        torch.save(model.state_dict(), "checkpoints/model_imdb.pt")
        
        test_texts, test_labels, test_d_ids = load_imdb("test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, tokenizer, batch_size=BATCH_SIZE)
        global_results["S1a"] = evaluate_model(model, test_loader, device, "S1a_Baseline_IMDb")
        save_results(global_results["S1a"], "results/results_s1a.json")

    # --- S1b: Monolingual Target Baseline (VSFC) ---
    if args.s in ["0", "1b"]:
        print_banner("Scenario 1b: Monolingual Target Baseline (VSFC)")
        t_all_vi, l_all_vi, d_all_vi = load_vsfc("train", max_samples=MAX_TRAIN)
        tv_train, tv_val, lv_train, lv_val, dv_train, dv_val = train_test_split(t_all_vi, l_all_vi, d_all_vi, test_size=0.1, random_state=42)
        
        train_loader_vi = make_dataloader(tv_train, lv_train, dv_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_vi = make_dataloader(tv_val, lv_val, dv_val, tokenizer, batch_size=BATCH_SIZE)
        
        model_vi = BaseModel(config["model"]["name"])
        weights_vi = compute_class_weights(lv_train)
        model_vi = train_model(model_vi, tokenizer, train_loader_vi, val_loader=val_loader_vi, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights_vi)
        
        test_texts_vi, test_labels_vi, test_d_ids_vi = load_vsfc("test", max_samples=MAX_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        global_results["S1b"] = evaluate_model(model_vi, test_loader_vi, device, "S1b_Baseline_VSFC")
        save_results(global_results["S1b"], "results/results_s1b.json")

    # --- S2: Zero-Shot Multilingual Transfer (IMDb -> VSFC) ---
    if args.s in ["0", "2"]:
        print_banner("Scenario 2: Zero-Shot Multilingual Transfer (IMDb -> VSFC)")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_imdb.pt"):
            model.load_state_dict(torch.load("checkpoints/model_imdb.pt", map_location=device))
        else:
            print("⚠️ Cần chạy S1a trước để có mô hình IMDb.")
        
        test_texts_vi, test_labels_vi, test_d_ids_vi = load_vsfc("test", max_samples=MAX_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        res_s2 = evaluate_model(model, test_loader_vi, device, "S2_ZeroShot_IMDb_VSFC")
        save_results(res_s2, "results/results_s2.json")

    # --- S3: Joint Multilingual Learning (IMDb + VSFC -> VSFC) ---
    if args.s in ["0", "3"]:
        print_banner("Scenario 3: Joint Multilingual Learning (IMDb + VSFC -> VSFC)")
        t_en, l_en, d_en = load_imdb("train", max_samples=MAX_TRAIN//2)
        t_vi, l_vi, d_vi = load_vsfc("train", max_samples=MAX_TRAIN//2)
        t_all, l_all, d_all = t_en + t_vi, l_en + l_vi, d_en + d_vi
        t_train, t_val, l_train, l_val, d_train, d_val = train_test_split(t_all, l_all, d_all, test_size=0.1, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, tokenizer, batch_size=BATCH_SIZE)
        
        model = BaseModel(config["model"]["name"])
        weights = compute_class_weights(l_train)
        model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
        
        test_vi_t, test_vi_l, test_vi_d = load_vsfc("test", max_samples=MAX_TEST)
        test_loader_vi = make_dataloader(test_vi_t, test_vi_l, test_vi_d, tokenizer, batch_size=BATCH_SIZE)
        res_s3 = evaluate_model(model, test_loader_vi, device, "S3_Joint_Multilingual_VSFC")
        save_results(res_s3, "results/results_s3.json")

    # --- S4: Zero-Shot Domain Transfer (IMDb -> Amazon) ---
    if args.s in ["0", "4"]:
        print_banner("Scenario 4: Zero-Shot Domain Transfer (IMDb -> Amazon)")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_imdb.pt"):
            model.load_state_dict(torch.load("checkpoints/model_imdb.pt", map_location=device))
        else:
            print("⚠️ Cần chạy S1a trước để có mô hình IMDb.")
        
        test_texts, test_labels, test_d_ids = load_amazon_split("english", "all", "test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s4 = evaluate_model(model, test_loader, device, "S4_ZeroShot_IMDb_Amazon")
        save_results(res_s4, "results/results_s4.json")

    # --- S5: Pure Multidomain Learning (IMDb + Yelp -> Amazon) ---
    if args.s in ["0", "5"]:
        print_banner("Scenario 5: Pure Multidomain Learning (IMDb + Yelp -> Amazon)")
        t1, l1, d1 = load_imdb("train", max_samples=MAX_TRAIN//2)
        t2, l2, d2 = load_yelp("train", max_samples=MAX_TRAIN//2)
        t_all, l_all, d_all = t1 + t2, l1 + l2, d1 + d2
        t_train, t_val, l_train, l_val, d_train, d_val = train_test_split(t_all, l_all, d_all, test_size=0.1, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, tokenizer, batch_size=BATCH_SIZE)
        
        model = BaseModel(config["model"]["name"])
        weights = compute_class_weights(l_train)
        model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
        
        test_texts, test_labels, test_d_ids = load_amazon_split("english", "all", "test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s5 = evaluate_model(model, test_loader, device, "S5_Pure_Multidomain_Amazon")
        save_results(res_s5, "results/results_s5.json")

    # --- S6: Domain Adaptation DANN (Source: IMDb, Target: Amazon) ---
    if args.s in ["0", "6"]:
        print_banner("Scenario 6: Domain Adaptation DANN (Source: IMDb, Target: Amazon)")
        s_texts, s_labels, s_d_ids = load_imdb("train", max_samples=MAX_TRAIN)
        t_texts, t_labels, t_d_ids = load_amazon_split("english", "all", "train", max_samples=MAX_TRAIN)
        
        s_train, s_val, l_train, l_val, d_train, d_val = train_test_split(s_texts, s_labels, s_d_ids, test_size=0.1, random_state=42)
        s_loader = make_dataloader(s_train, l_train, d_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(s_val, l_val, d_val, tokenizer, batch_size=BATCH_SIZE)
        t_loader = make_dataloader(t_texts, t_labels, t_d_ids, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        
        model_dann = DANNModel(config["model"]["name"])
        weights = compute_class_weights(l_train)
        model_dann = train_dann(model_dann, tokenizer, s_loader, t_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
        
        test_texts, test_labels, test_d_ids = load_amazon_split("english", "all", "test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s6 = evaluate_model(model_dann, test_loader, device, "S6_DANN_Amazon")
        save_results(res_s6, "results/results_s6.json")

    # --- S7: Supervised Target Upper Bound (Amazon -> Amazon) ---
    if args.s in ["0", "7"]:
        print_banner("Scenario 7: Supervised Target Upper Bound (Amazon -> Amazon)")
        t_all, l_all, d_all = load_amazon_split("english", "all", "train", max_samples=MAX_TRAIN)
        t_train, t_val, l_train, l_val, d_train, d_val = train_test_split(t_all, l_all, d_all, test_size=0.1, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, tokenizer, batch_size=BATCH_SIZE)
        
        model = BaseModel(config["model"]["name"])
        weights = compute_class_weights(l_train)
        model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
        
        test_texts, test_labels, test_d_ids = load_amazon_split("english", "all", "test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s7 = evaluate_model(model, test_loader, device, "S7_UpperBound_Amazon")
        save_results(res_s7, "results/results_s7.json")

    print_banner("ALL EXPERIMENTS COMPLETED")
    try:
        generate_aggregate_report()
    except Exception as e:
        print(f"⚠️ Không thể tạo báo cáo tổng hợp: {e}")

if __name__ == "__main__":
    main()
