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
from src.model import BaseModel, DANNModel, AdvancedMultiTaskModel
from src.train import train_model, train_dann, train_multitask, compute_class_weights
from src.evaluate import evaluate_model
from src.visualize_embeddings import visualize_tsne
from src.report_generator import generate_aggregate_report
from src.utils import print_banner, save_results, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=str, default="0", help="Scenario to run (0=all, 1a, 1b, 2, 3, 4, 5, 6a, 6b, 7, 8, 9, 10, 11, 12, 13, 14)")
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    BASE_TEST = config["scenarios"].get("base_samples_test", 2000)
    BASE_TRAIN = config["scenarios"].get("base_samples_train", 5000)
    FEW_SHOT = config["scenarios"].get("few_shot_samples", 500)
    
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = int(config["training"]["epochs"])
    LR = float(config["training"]["learning_rate"])
    
    global_results = {}
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    # --- S1a: Monolingual Source Baseline (IMDb) ---
    if args.s in ["0", "1a"] and config["scenarios"].get("run_s1", True):
        print_banner("Scenario 1a: Monolingual Source Baseline (IMDb)")
        t_all, l_all, d_all, la_all = load_imdb("train", max_samples=BASE_TRAIN)
        t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.2, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        
        model = BaseModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_imdb.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            weights = compute_class_weights(l_train)
            model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
            torch.save(model.state_dict(), checkpoint_path)
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_imdb("test", max_samples=BASE_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        global_results["S1a"] = evaluate_model(model, test_loader, device, "S1a_Baseline_IMDb")
        save_results(global_results["S1a"], "results/results_s1a.json")

    # --- S1b: Monolingual Target Baseline (VSFC) ---
    if args.s in ["0", "1b"] and config["scenarios"].get("run_s1", True):
        print_banner("Scenario 1b: Monolingual Target Baseline (VSFC)")
        t_all_vi, l_all_vi, d_all_vi, la_all_vi = load_vsfc("train", max_samples=BASE_TRAIN)
        tv_train, tv_val, lv_train, lv_val, dv_train, dv_val, lav_train, lav_val = train_test_split(t_all_vi, l_all_vi, d_all_vi, la_all_vi, test_size=0.2, random_state=42)
        
        train_loader_vi = make_dataloader(tv_train, lv_train, dv_train, lav_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_vi = make_dataloader(tv_val, lv_val, dv_val, lav_val, tokenizer, batch_size=BATCH_SIZE)
        
        model_vi = BaseModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_vsfc.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model_vi.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            weights_vi = compute_class_weights(lv_train)
            model_vi = train_model(model_vi, tokenizer, train_loader_vi, val_loader=val_loader_vi, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights_vi)
            torch.save(model_vi.state_dict(), checkpoint_path)
        
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        global_results["S1b"] = evaluate_model(model_vi, test_loader_vi, device, "S1b_Baseline_VSFC")
        save_results(global_results["S1b"], "results/results_s1b.json")

    # --- S2: Zero-Shot Multilingual Transfer (IMDb -> VSFC) ---
    if args.s in ["0", "2"] and config["scenarios"].get("run_s2", True):
        print_banner("Scenario 2: Zero-Shot Multilingual Transfer (IMDb -> VSFC)")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_imdb.pt"):
            model.load_state_dict(torch.load("checkpoints/model_imdb.pt", map_location=device))
        else:
            print("⚠️ Cần chạy S1a trước để có mô hình IMDb.")
        
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        res_s2 = evaluate_model(model, test_loader_vi, device, "S2_ZeroShot_IMDb_VSFC")
        save_results(res_s2, "results/results_s2.json")
        
        try:
            vis_en_t, vis_en_l, vis_en_d, vis_en_la = load_imdb("test", max_samples=300)
            vis_vi_t, vis_vi_l, vis_vi_d, vis_vi_la = load_vsfc("test", max_samples=300)
            ld_en = make_dataloader(vis_en_t, vis_en_l, vis_en_d, vis_en_la, tokenizer, batch_size=BATCH_SIZE)
            ld_vi = make_dataloader(vis_vi_t, vis_vi_l, vis_vi_d, vis_vi_la, tokenizer, batch_size=BATCH_SIZE)
            visualize_tsne(model, tokenizer, [ld_en, ld_vi], ["English (IMDb)", "Vietnamese (VSFC)"], device, "S2_Language_Gap_ZeroShot")
        except Exception as e:
            pass

    # --- S3: Joint Multilingual Learning (IMDb + VSFC) ---
    if args.s in ["0", "3"] and config["scenarios"].get("run_s3", True):
        print_banner("Scenario 3: Joint Multilingual Learning (IMDb + VSFC)")
        t_en, l_en, d_en, la_en = load_imdb("train", max_samples=BASE_TRAIN)
        t_vi, l_vi, d_vi, la_vi = load_vsfc("train", max_samples=BASE_TRAIN)
        t_all, l_all, d_all, la_all = t_en + t_vi, l_en + l_vi, d_en + d_vi, la_en + la_vi
        t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.2, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        
        model = BaseModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_s3_joint.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            weights = compute_class_weights(l_train)
            model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
            torch.save(model.state_dict(), checkpoint_path)
        
        print("\n--- S3a: Testing on Vietnamese (VSFC) ---")
        test_vi_t, test_vi_l, test_vi_d, test_vi_la = load_vsfc("test", max_samples=BASE_TEST)
        test_loader_vi = make_dataloader(test_vi_t, test_vi_l, test_vi_d, test_vi_la, tokenizer, batch_size=BATCH_SIZE)
        res_s3a = evaluate_model(model, test_loader_vi, device, "S3a_Joint_Multilingual_VSFC")
        save_results(res_s3a, "results/results_s3a.json")

        print("\n--- S3b: Testing on English (IMDb) ---")
        test_en_t, test_en_l, test_en_d, test_en_la = load_imdb("test", max_samples=BASE_TEST)
        test_loader_en = make_dataloader(test_en_t, test_en_l, test_en_d, test_en_la, tokenizer, batch_size=BATCH_SIZE)
        res_s3b = evaluate_model(model, test_loader_en, device, "S3b_Joint_Multilingual_IMDb")
        save_results(res_s3b, "results/results_s3b.json")

    # --- S4: Zero-Shot Domain Transfer (IMDb -> Amazon) ---
    if args.s in ["0", "4"] and config["scenarios"].get("run_s4", True):
        print_banner("Scenario 4: Zero-Shot Domain Transfer (IMDb -> Amazon)")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_imdb.pt"):
            model.load_state_dict(torch.load("checkpoints/model_imdb.pt", map_location=device))
        else:
            print("⚠️ Cần chạy S1a trước.")
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s4 = evaluate_model(model, test_loader, device, "S4_ZeroShot_IMDb_Amazon")
        save_results(res_s4, "results/results_s4.json")

    # --- S5: Pure Multidomain Learning (IMDb + Yelp -> Amazon) ---
    if args.s in ["0", "5"] and config["scenarios"].get("run_s5", True):
        print_banner("Scenario 5: Pure Multidomain Learning (IMDb + Yelp -> Amazon)")
        t1, l1, d1, la1 = load_imdb("train", max_samples=BASE_TRAIN)
        t2, l2, d2, la2 = load_yelp("train", max_samples=BASE_TRAIN)
        t_all, l_all, d_all, la_all = t1 + t2, l1 + l2, d1 + d2, la1 + la2
        t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.2, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        
        model = BaseModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_s5_multidomain.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            weights = compute_class_weights(l_train)
            model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
            torch.save(model.state_dict(), checkpoint_path)
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s5 = evaluate_model(model, test_loader, device, "S5_Pure_Multidomain_Amazon")
        save_results(res_s5, "results/results_s5.json")

    # --- S6a: Single-Source Domain Adaptation DANN (Source: IMDb, Target: Amazon) ---
    if args.s in ["0", "6", "6a"] and config["scenarios"].get("run_s6", True):
        print_banner("Scenario 6a: Single-Source DANN (Source: IMDb, Target: Amazon)")
        s_texts, s_labels, s_d_ids, s_la_ids = load_imdb("train", max_samples=BASE_TRAIN)
        t_texts, t_labels, t_d_ids, t_la_ids = load_amazon_split("english", "all", "train", max_samples=BASE_TRAIN, unlabeled=True)
        
        s_train, s_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(s_texts, s_labels, s_d_ids, s_la_ids, test_size=0.2, random_state=42)
        s_loader = make_dataloader(s_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(s_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        t_loader = make_dataloader(t_texts, t_labels, t_d_ids, t_la_ids, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        
        model_dann = DANNModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_dann_s6a.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model_dann.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            if os.path.exists("checkpoints/model_imdb.pt"):
                model_dann.load_state_dict(torch.load("checkpoints/model_imdb.pt", map_location=device), strict=False)
            
            weights = compute_class_weights(l_train)
            model_dann = train_dann(model_dann, tokenizer, s_loader, t_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR/10.0, device=device, class_weights=weights)
            torch.save(model_dann.state_dict(), checkpoint_path)
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s6a = evaluate_model(model_dann, test_loader, device, "S6a_DANN_Amazon")
        save_results(res_s6a, "results/results_s6a.json")

    # --- S6b: Multi-Source Domain Adaptation DANN (Source: IMDb + Yelp, Target: Amazon) ---
    if args.s in ["0", "6", "6b"] and config["scenarios"].get("run_s6", True):
        print_banner("Scenario 6b: Multi-Source DANN (Source: IMDb+Yelp, Target: Amazon)")
        t1, l1, d1, la1 = load_imdb("train", max_samples=BASE_TRAIN)
        t2, l2, d2, la2 = load_yelp("train", max_samples=BASE_TRAIN)
        s_texts, s_labels, s_d_ids, s_la_ids = t1 + t2, l1 + l2, d1 + d2, la1 + la2
        # Target needs 2*BASE_TRAIN to match source
        t_texts, t_labels, t_d_ids, t_la_ids = load_amazon_split("english", "all", "train", max_samples=BASE_TRAIN * 2, unlabeled=True)
        
        s_train, s_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(s_texts, s_labels, s_d_ids, s_la_ids, test_size=0.2, random_state=42)
        s_loader = make_dataloader(s_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(s_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        t_loader = make_dataloader(t_texts, t_labels, t_d_ids, t_la_ids, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        
        model_dann = DANNModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_dann_s6b.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model_dann.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            if os.path.exists("checkpoints/model_s5_multidomain.pt"):
                model_dann.load_state_dict(torch.load("checkpoints/model_s5_multidomain.pt", map_location=device), strict=False)
            
            weights = compute_class_weights(l_train)
            model_dann = train_dann(model_dann, tokenizer, s_loader, t_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR/10.0, device=device, class_weights=weights)
            torch.save(model_dann.state_dict(), checkpoint_path)
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s6b = evaluate_model(model_dann, test_loader, device, "S6b_MultiSource_DANN_Amazon")
        save_results(res_s6b, "results/results_s6b.json")

    # --- S7: Supervised Target Upper Bound (Amazon -> Amazon) ---
    if args.s in ["0", "7"] and config["scenarios"].get("run_s7", True):
        print_banner("Scenario 7: Supervised Target Upper Bound (Amazon -> Amazon)")
        t_all, l_all, d_all, la_all = load_amazon_split("english", "all", "train", max_samples=BASE_TRAIN)
        t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.2, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        
        model = BaseModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_upper_bound_s7.pt"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            weights = compute_class_weights(l_train)
            model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
            torch.save(model.state_dict(), checkpoint_path)
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s7 = evaluate_model(model, test_loader, device, "S7_UpperBound_Amazon")
        save_results(res_s7, "results/results_s7.json")

    # --- S8: Multi-domain Evaluation (Amz+IMDb+Yelp -> Each) ---
    if args.s in ["0", "8"] and config["scenarios"].get("run_s8", True):
        print_banner("Scenario 8: Unified Multi-domain Learning (Amz+IMDb+Yelp)")
        t1, l1, d1, la1 = load_amazon_split("english", "all", "train", max_samples=BASE_TRAIN)
        t2, l2, d2, la2 = load_imdb("train", max_samples=BASE_TRAIN)
        t3, l3, d3, la3 = load_yelp("train", max_samples=BASE_TRAIN)
        
        t_all, l_all, d_all, la_all = t1+t2+t3, l1+l2+l3, d1+d2+d3, la1+la2+la3
        t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.2, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        
        model = BaseModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_s8_mdl.pt"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            weights = compute_class_weights(l_train)
            model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
            torch.save(model.state_dict(), checkpoint_path)
        
        for domain_name, loader_func in [("Amazon", load_amazon_split), ("IMDb", load_imdb), ("Yelp", load_yelp)]:
            if domain_name == "Amazon":
                tt, tl, td, tla = loader_func("english", "all", "test", max_samples=BASE_TEST)
            else:
                tt, tl, td, tla = loader_func("test", max_samples=BASE_TEST)
            
            t_loader = make_dataloader(tt, tl, td, tla, tokenizer, batch_size=BATCH_SIZE)
            res = evaluate_model(model, t_loader, device, f"S8_MDL_{domain_name}")
            save_results(res, f"results/results_s8_{domain_name.lower()}.json")

    # --- S9: Supervised Fine-tuning (SFT) ---
    if args.s in ["0", "9", "9a", "9b"] and config["scenarios"].get("run_s9", True):
        # S9a: Single-Source (IMDb) -> Amazon SFT
        if args.s in ["0", "9", "9a"]:
            print_banner(f"Scenario 9a: Single-Source SFT (IMDb -> {FEW_SHOT} Amazon)")
            model = BaseModel(config["model"]["name"])
            if os.path.exists("checkpoints/model_imdb.pt"):
                model.load_state_dict(torch.load("checkpoints/model_imdb.pt", map_location=device))
            
            t_amz, l_amz, d_amz, la_amz = load_amazon_split("english", "all", "train", max_samples=FEW_SHOT)
            train_loader = make_dataloader(t_amz, l_amz, d_amz, la_amz, tokenizer, batch_size=8, shuffle=True)
            
            checkpoint_path = "checkpoints/model_sft_s9a.pt"
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            else:
                model = train_model(model, tokenizer, train_loader, num_epochs=3, lr=5e-6, device=device)
                torch.save(model.state_dict(), checkpoint_path)
            
            tt, tl, td, tla = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
            test_loader = make_dataloader(tt, tl, td, tla, tokenizer, batch_size=BATCH_SIZE)
            res_s9a = evaluate_model(model, test_loader, device, "S9a_SFT_IMDb_Amazon")
            save_results(res_s9a, "results/results_s9a.json")

        # S9b: Multi-Source (IMDb + Yelp) -> Amazon SFT
        if args.s in ["0", "9", "9b"]:
            print_banner(f"Scenario 9b: Multi-Source SFT (IMDb+Yelp -> {FEW_SHOT} Amazon)")
            model = BaseModel(config["model"]["name"])
            if os.path.exists("checkpoints/model_s5_multidomain.pt"):
                model.load_state_dict(torch.load("checkpoints/model_s5_multidomain.pt", map_location=device))
            
            t_amz, l_amz, d_amz, la_amz = load_amazon_split("english", "all", "train", max_samples=FEW_SHOT)
            train_loader = make_dataloader(t_amz, l_amz, d_amz, la_amz, tokenizer, batch_size=8, shuffle=True)
            
            checkpoint_path = "checkpoints/model_sft_s9b.pt"
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            else:
                model = train_model(model, tokenizer, train_loader, num_epochs=3, lr=5e-6, device=device)
                torch.save(model.state_dict(), checkpoint_path)
            
            tt, tl, td, tla = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
            test_loader = make_dataloader(tt, tl, td, tla, tokenizer, batch_size=BATCH_SIZE)
            res_s9b = evaluate_model(model, test_loader, device, "S9b_SFT_Multi_Amazon")
            save_results(res_s9b, "results/results_s9b.json")

    # --- S10a: Multi-source Cross-lingual Transfer (IMDb + Yelp -> VSFC) ---
    if args.s in ["0", "10", "10a"] and config["scenarios"].get("run_s10", True):
        print_banner("Scenario 10a: Multi-source Cross-lingual Transfer (IMDb+Yelp -> VSFC)")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_s5_multidomain.pt"):
            model.load_state_dict(torch.load("checkpoints/model_s5_multidomain.pt", map_location=device))
        
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        res_s10a = evaluate_model(model, test_loader_vi, device, "S10a_MultiCross_ZeroShot")
        save_results(res_s10a, "results/results_s10a.json")

    # --- S10b: Multi-source Cross-lingual DANN (IMDb + Yelp -> VSFC) ---
    if args.s in ["0", "10", "10b"] and config["scenarios"].get("run_s10", True):
        print_banner("Scenario 10b: Multi-source Cross-lingual DANN (IMDb+Yelp -> VSFC)")
        t1, l1, d1, la1 = load_imdb("train", max_samples=BASE_TRAIN)
        t2, l2, d2, la2 = load_yelp("train", max_samples=BASE_TRAIN)
        s_texts, s_labels, s_d_ids, s_la_ids = t1 + t2, l1 + l2, d1 + d2, la1 + la2
        t_texts, t_labels, t_d_ids, t_la_ids = load_vsfc("train", max_samples=BASE_TRAIN * 2, unlabeled=True)
        
        s_train, s_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(s_texts, s_labels, s_d_ids, s_la_ids, test_size=0.2, random_state=42)
        s_loader = make_dataloader(s_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(s_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        t_loader = make_dataloader(t_texts, t_labels, t_d_ids, t_la_ids, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        
        model_dann = DANNModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_dann_s10b.pt"
        if os.path.exists(checkpoint_path):
            model_dann.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            if os.path.exists("checkpoints/model_s5_multidomain.pt"):
                model_dann.load_state_dict(torch.load("checkpoints/model_s5_multidomain.pt", map_location=device), strict=False)
            
            weights = compute_class_weights(l_train)
            model_dann = train_dann(model_dann, tokenizer, s_loader, t_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR/10.0, device=device, class_weights=weights)
            torch.save(model_dann.state_dict(), checkpoint_path)
        
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        res_s10b = evaluate_model(model_dann, test_loader_vi, device, "S10b_MultiCross_DANN")
        save_results(res_s10b, "results/results_s10b.json")

    # --- S11: Model Comparison (mBERT vs XLM-R) ---
    if args.s in ["0", "11", "11a", "11b", "11c", "11d"] and config["scenarios"].get("run_s11", True):
        mbert_name = "bert-base-multilingual-cased"
        tokenizer_mbert = AutoTokenizer.from_pretrained(mbert_name)

        model_mbert_base = BaseModel(mbert_name)
        checkpoint_base = "checkpoints/model_mbert_imdb.pt"
        
        # S11_base / S11c / S11d require a trained base mBERT on IMDb
        if args.s in ["0", "11", "11c", "11d"]:
            if os.path.exists(checkpoint_base):
                model_mbert_base.load_state_dict(torch.load(checkpoint_base, map_location=device))
            else:
                print_banner("Scenario 11_base: Train mBERT on IMDb for Zero-Shot Comparison")
                t_all, l_all, d_all, la_all = load_imdb("train", max_samples=BASE_TRAIN)
                t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.2, random_state=42)
                
                train_loader = make_dataloader(t_train, l_train, d_train, la_train, tokenizer_mbert, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = make_dataloader(t_val, l_val, d_val, la_val, tokenizer_mbert, batch_size=BATCH_SIZE)
                
                weights = compute_class_weights(l_train)
                model_mbert_base = train_model(model_mbert_base, tokenizer_mbert, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
                torch.save(model_mbert_base.state_dict(), checkpoint_base)

        if args.s in ["0", "11", "11c"]:
            print_banner("Scenario 11c: mBERT on S4 task (Zero-Shot IMDb -> Amazon)")
            test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
            test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer_mbert, batch_size=BATCH_SIZE)
            res_s11c = evaluate_model(model_mbert_base, test_loader, device, "S11c_ModelComp_mBERT_ZeroShot_Amazon")
            save_results(res_s11c, "results/results_s11c.json")

        if args.s in ["0", "11", "11d"]:
            print_banner("Scenario 11d: mBERT on S2 task (Zero-Shot IMDb -> VSFC)")
            test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
            test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer_mbert, batch_size=BATCH_SIZE)
            res_s11d = evaluate_model(model_mbert_base, test_loader_vi, device, "S11d_ModelComp_mBERT_ZeroShot_VSFC")
            save_results(res_s11d, "results/results_s11d.json")

        if args.s in ["0", "11", "11a"]:
            print_banner("Scenario 11a: mBERT on S6b task (IMDb+Yelp -> Amazon DANN)")
            t1, l1, d1, la1 = load_imdb("train", max_samples=BASE_TRAIN)
            t2, l2, d2, la2 = load_yelp("train", max_samples=BASE_TRAIN)
            s_texts, s_labels, s_d_ids, s_la_ids = t1 + t2, l1 + l2, d1 + d2, la1 + la2
            t_texts, t_labels, t_d_ids, t_la_ids = load_amazon_split("english", "all", "train", max_samples=BASE_TRAIN * 2, unlabeled=True)
            
            s_train, s_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(s_texts, s_labels, s_d_ids, s_la_ids, test_size=0.2, random_state=42)
            s_loader = make_dataloader(s_train, l_train, d_train, la_train, tokenizer_mbert, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = make_dataloader(s_val, l_val, d_val, la_val, tokenizer_mbert, batch_size=BATCH_SIZE)
            t_loader = make_dataloader(t_texts, t_labels, t_d_ids, t_la_ids, tokenizer_mbert, batch_size=BATCH_SIZE, shuffle=True)
            
            model_mbert = DANNModel(mbert_name)
            checkpoint_path = "checkpoints/model_dann_s11a_mbert.pt"
            if os.path.exists(checkpoint_path):
                model_mbert.load_state_dict(torch.load(checkpoint_path, map_location=device))
            else:
                if os.path.exists(checkpoint_base):
                    model_mbert.load_state_dict(torch.load(checkpoint_base, map_location=device), strict=False)
                weights = compute_class_weights(l_train)
                model_mbert = train_dann(model_mbert, tokenizer_mbert, s_loader, t_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR/10.0, device=device, class_weights=weights)
                torch.save(model_mbert.state_dict(), checkpoint_path)
                
            test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
            test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer_mbert, batch_size=BATCH_SIZE)
            res_s11a = evaluate_model(model_mbert, test_loader, device, "S11a_ModelComp_mBERT_Amazon")
            save_results(res_s11a, "results/results_s11a.json")

        if args.s in ["0", "11", "11b"]:
            print_banner("Scenario 11b: mBERT on S10b task (IMDb+Yelp -> VSFC DANN)")
            t1, l1, d1, la1 = load_imdb("train", max_samples=BASE_TRAIN)
            t2, l2, d2, la2 = load_yelp("train", max_samples=BASE_TRAIN)
            s_texts, s_labels, s_d_ids, s_la_ids = t1 + t2, l1 + l2, d1 + d2, la1 + la2
            t_texts_vi, t_labels_vi, t_d_ids_vi, t_la_ids_vi = load_vsfc("train", max_samples=BASE_TRAIN * 2, unlabeled=True)
            
            s_train, s_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(s_texts, s_labels, s_d_ids, s_la_ids, test_size=0.2, random_state=42)
            s_loader = make_dataloader(s_train, l_train, d_train, la_train, tokenizer_mbert, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = make_dataloader(s_val, l_val, d_val, la_val, tokenizer_mbert, batch_size=BATCH_SIZE)
            t_loader_vi = make_dataloader(t_texts_vi, t_labels_vi, t_d_ids_vi, t_la_ids_vi, tokenizer_mbert, batch_size=BATCH_SIZE, shuffle=True)
            
            model_mbert_vi = DANNModel(mbert_name)
            checkpoint_path = "checkpoints/model_dann_s11b_mbert.pt"
            if os.path.exists(checkpoint_path):
                model_mbert_vi.load_state_dict(torch.load(checkpoint_path, map_location=device))
            else:
                if os.path.exists(checkpoint_base):
                    model_mbert_vi.load_state_dict(torch.load(checkpoint_base, map_location=device), strict=False)
                weights = compute_class_weights(l_train)
                model_mbert_vi = train_dann(model_mbert_vi, tokenizer_mbert, s_loader, t_loader_vi, val_loader=val_loader, num_epochs=EPOCHS, lr=LR/10.0, device=device, class_weights=weights)
                torch.save(model_mbert_vi.state_dict(), checkpoint_path)
                
            test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
            test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer_mbert, batch_size=BATCH_SIZE)
            res_s11b = evaluate_model(model_mbert_vi, test_loader_vi, device, "S11b_ModelComp_mBERT_VSFC")
            save_results(res_s11b, "results/results_s11b.json")

    # --- S12: Cross-lingual Target Fine-Tuning (IMDb -> VSFC) ---
    if args.s in ["0", "12"]:
        print_banner(f"Scenario 12: Cross-lingual Target Fine-Tuning (IMDb -> {FEW_SHOT} VSFC)")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_imdb.pt"):
            model.load_state_dict(torch.load("checkpoints/model_imdb.pt", map_location=device))
        
        t_vsfc, l_vsfc, d_vsfc, la_vsfc = load_vsfc("train", max_samples=FEW_SHOT)
        train_loader = make_dataloader(t_vsfc, l_vsfc, d_vsfc, la_vsfc, tokenizer, batch_size=8, shuffle=True)
        
        checkpoint_path = "checkpoints/model_sft_s12.pt"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            model = train_model(model, tokenizer, train_loader, num_epochs=3, lr=5e-6, device=device)
            torch.save(model.state_dict(), checkpoint_path)
        
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        res_s12 = evaluate_model(model, test_loader_vi, device, "S12_SFT_IMDb_VSFC")
        save_results(res_s12, "results/results_s12.json")

    # --- S13: Translation-Based Cross-lingual Methods ---
    if args.s in ["0", "13"]:
        print_banner("Scenario 13: Translation-Based Methods (VSFC -> English -> IMDb Model)")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_imdb.pt"):
            model.load_state_dict(torch.load("checkpoints/model_imdb.pt", map_location=device))
            
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
        
        print("🌍 Đang dịch dữ liệu test từ Tiếng Việt sang Tiếng Anh bằng deep-translator...")
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='vi', target='en')
            test_texts_translated = []
            from tqdm import tqdm
            for text in tqdm(test_texts_vi, desc="Translating"):
                try:
                    trans = translator.translate(text[:4999])
                    test_texts_translated.append(trans if trans else "")
                except Exception as e:
                    test_texts_translated.append(text)
        except ImportError:
            test_texts_translated = test_texts_vi
            
        test_loader_trans = make_dataloader(test_texts_translated, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        res_s13 = evaluate_model(model, test_loader_trans, device, "S13_Translation_VSFC_EN")
        save_results(res_s13, "results/results_s13.json")

    # --- S14: Advanced Multi-task Learning (Sentiment + Domain + Language) ---
    if args.s in ["0", "14"]:
        print_banner("Scenario 14: Unified Multi-task Learning Framework (S+D+L)")
        t_en1, l_en1, d_en1, la_en1 = load_imdb("train", max_samples=BASE_TRAIN)
        t_en2, l_en2, d_en2, la_en2 = load_yelp("train", max_samples=BASE_TRAIN)
        t_amz, l_amz, d_amz, la_amz = load_amazon_split("english", "all", "train", max_samples=BASE_TRAIN)
        
        # Lấy dữ liệu tiếng Việt (VSFC) Unlabeled cho Cross-lingual target
        t_vi, l_vi, d_vi, la_vi = load_vsfc("train", max_samples=BASE_TRAIN, unlabeled=True)
        
        t_all = t_en1 + t_en2 + t_amz + t_vi
        l_all = l_en1 + l_en2 + l_amz + l_vi
        d_all = d_en1 + d_en2 + d_amz + d_vi
        la_all = la_en1 + la_en2 + la_amz + la_vi
        
        s_train, s_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.1, random_state=42)
        
        train_loader = make_dataloader(s_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(s_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        
        model_mtl = AdvancedMultiTaskModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_s14_multitask.pt"
        if os.path.exists(checkpoint_path):
            model_mtl.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            if os.path.exists("checkpoints/model_s8_mdl.pt"):
                model_mtl.load_state_dict(torch.load("checkpoints/model_s8_mdl.pt", map_location=device), strict=False)
            
            weights = compute_class_weights([lbl for lbl in l_train if lbl >= 0])
            model_mtl = train_multitask(model_mtl, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR/10.0, device=device, class_weights=weights)
            torch.save(model_mtl.state_dict(), checkpoint_path)
            
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        
        model_mtl.eval()
        model_mtl.to(device)
        correct, total = 0, 0
        from sklearn.metrics import classification_report
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for b in test_loader_vi:
                s_lgt, _, _ = model_mtl(b["input_ids"].to(device), b["attention_mask"].to(device))
                preds = torch.argmax(s_lgt, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(b["labels"].cpu().numpy())
                correct += (preds == b["labels"].to(device)).sum().item()
                total += b["labels"].size(0)
        
        acc = correct / total
        print(f"\n[S14] Accuracy: {acc*100:.2f}%")
        
        res_s14 = {
            "scenario": "S14_Advanced_MultiTask",
            "accuracy": acc,
            "report": classification_report(all_labels, all_preds, output_dict=True)
        }
        save_results(res_s14, "results/results_s14.json")

    print_banner("ALL EXPERIMENTS COMPLETED")
    try:
        generate_aggregate_report()
    except Exception as e:
        print(f"⚠️ Không thể tạo báo cáo tổng hợp: {e}")

if __name__ == "__main__":
    main()
