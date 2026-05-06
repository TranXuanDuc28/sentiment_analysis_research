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
from src.visualize_embeddings import visualize_tsne
from src.report_generator import generate_aggregate_report
from src.utils import print_banner, save_results, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=str, default="0", help="Scenario to run (0=all, 1a, 1b, 2, 3, 4, 5, 6a, 6b, 7, 8, 9)")
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
        t_all, l_all, d_all, la_all = load_imdb("train", max_samples=MAX_TRAIN)
        t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.1, random_state=42)
        
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
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_imdb("test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        global_results["S1a"] = evaluate_model(model, test_loader, device, "S1a_Baseline_IMDb")
        save_results(global_results["S1a"], "results/results_s1a.json")

    # --- S1b: Monolingual Target Baseline (VSFC) ---
    if args.s in ["0", "1b"]:
        print_banner("Scenario 1b: Monolingual Target Baseline (VSFC)")
        t_all_vi, l_all_vi, d_all_vi, la_all_vi = load_vsfc("train", max_samples=MAX_TRAIN)
        tv_train, tv_val, lv_train, lv_val, dv_train, dv_val, lav_train, lav_val = train_test_split(t_all_vi, l_all_vi, d_all_vi, la_all_vi, test_size=0.1, random_state=42)
        
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
        
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=MAX_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
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
        
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=MAX_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        res_s2 = evaluate_model(model, test_loader_vi, device, "S2_ZeroShot_IMDb_VSFC")
        save_results(res_s2, "results/results_s2.json")
        
        # Visualize Language Gap
        try:
            vis_en_t, vis_en_l, vis_en_d, vis_en_la = load_imdb("test", max_samples=300)
            vis_vi_t, vis_vi_l, vis_vi_d, vis_vi_la = load_vsfc("test", max_samples=300)
            ld_en = make_dataloader(vis_en_t, vis_en_l, vis_en_d, vis_en_la, tokenizer, batch_size=BATCH_SIZE)
            ld_vi = make_dataloader(vis_vi_t, vis_vi_l, vis_vi_d, vis_vi_la, tokenizer, batch_size=BATCH_SIZE)
            visualize_tsne(model, tokenizer, [ld_en, ld_vi], ["English (IMDb)", "Vietnamese (VSFC)"], device, "S2_Language_Gap_ZeroShot")
        except Exception as e:
            print(f"⚠️ Không thể tạo biểu đồ t-SNE: {e}")

    # --- S3: Joint Multilingual Learning (IMDb + VSFC) ---
    if args.s in ["0", "3"]:
        print_banner("Scenario 3: Joint Multilingual Learning (IMDb + VSFC)")
        t_en, l_en, d_en, la_en = load_imdb("train", max_samples=MAX_TRAIN//2)
        t_vi, l_vi, d_vi, la_vi = load_vsfc("train", max_samples=MAX_TRAIN//2)
        t_all, l_all, d_all, la_all = t_en + t_vi, l_en + l_vi, d_en + d_vi, la_en + la_vi
        t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.1, random_state=42)
        
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
        
        # Test 3a: On Vietnamese
        print("\n--- S3a: Testing on Vietnamese (VSFC) ---")
        test_vi_t, test_vi_l, test_vi_d, test_vi_la = load_vsfc("test", max_samples=MAX_TEST)
        test_loader_vi = make_dataloader(test_vi_t, test_vi_l, test_vi_d, test_vi_la, tokenizer, batch_size=BATCH_SIZE)
        res_s3a = evaluate_model(model, test_loader_vi, device, "S3a_Joint_Multilingual_VSFC")
        save_results(res_s3a, "results/results_s3a.json")

        # Test 3b: On English
        print("\n--- S3b: Testing on English (IMDb) ---")
        test_en_t, test_en_l, test_en_d, test_en_la = load_imdb("test", max_samples=MAX_TEST)
        test_loader_en = make_dataloader(test_en_t, test_en_l, test_en_d, test_en_la, tokenizer, batch_size=BATCH_SIZE)
        res_s3b = evaluate_model(model, test_loader_en, device, "S3b_Joint_Multilingual_IMDb")
        save_results(res_s3b, "results/results_s3b.json")
        
        # Visualize Multilingual Alignment
        try:
            vis_en_t, vis_en_l, vis_en_d, vis_en_la = load_imdb("test", max_samples=300)
            vis_vi_t, vis_vi_l, vis_vi_d, vis_vi_la = load_vsfc("test", max_samples=300)
            ld_en = make_dataloader(vis_en_t, vis_en_l, vis_en_d, vis_en_la, tokenizer, batch_size=BATCH_SIZE)
            ld_vi = make_dataloader(vis_vi_t, vis_vi_l, vis_vi_d, vis_vi_la, tokenizer, batch_size=BATCH_SIZE)
            visualize_tsne(model, tokenizer, [ld_en, ld_vi], ["English (IMDb)", "Vietnamese (VSFC)"], device, "S3_Multilingual_Alignment")
        except Exception as e:
            print(f"⚠️ Không thể tạo biểu đồ t-SNE: {e}")

    # --- S4: Zero-Shot Domain Transfer (IMDb -> Amazon) ---
    if args.s in ["0", "4"]:
        print_banner("Scenario 4: Zero-Shot Domain Transfer (IMDb -> Amazon)")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_imdb.pt"):
            model.load_state_dict(torch.load("checkpoints/model_imdb.pt", map_location=device))
        else:
            print("⚠️ Cần chạy S1a trước để có mô hình IMDb.")
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s4 = evaluate_model(model, test_loader, device, "S4_ZeroShot_IMDb_Amazon")
        save_results(res_s4, "results/results_s4.json")
        
        # Visualize Domain Gap (Before DANN)
        try:
            vis_src_t, vis_src_l, vis_src_d, vis_src_la = load_imdb("test", max_samples=300)
            vis_tgt_t, vis_tgt_l, vis_tgt_d, vis_tgt_la = load_amazon_split("english", "all", "test", max_samples=300)
            ld_src = make_dataloader(vis_src_t, vis_src_l, vis_src_d, vis_src_la, tokenizer, batch_size=BATCH_SIZE)
            ld_tgt = make_dataloader(vis_tgt_t, vis_tgt_l, vis_tgt_d, vis_tgt_la, tokenizer, batch_size=BATCH_SIZE)
            visualize_tsne(model, tokenizer, [ld_src, ld_tgt], ["Source (IMDb)", "Target (Amazon)"], device, "S4_Domain_Gap_Before_DANN")
        except Exception as e:
            print(f"⚠️ Không thể tạo biểu đồ t-SNE: {e}")

    # --- S5: Pure Multidomain Learning (IMDb + Yelp -> Amazon) ---
    if args.s in ["0", "5"]:
        print_banner("Scenario 5: Pure Multidomain Learning (IMDb + Yelp -> Amazon)")
        t1, l1, d1, la1 = load_imdb("train", max_samples=MAX_TRAIN//2)
        t2, l2, d2, la2 = load_yelp("train", max_samples=MAX_TRAIN//2)
        t_all, l_all, d_all, la_all = t1 + t2, l1 + l2, d1 + d2, la1 + la2
        t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.1, random_state=42)
        
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
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s5 = evaluate_model(model, test_loader, device, "S5_Pure_Multidomain_Amazon")
        save_results(res_s5, "results/results_s5.json")
        
        # Visualize Domain Gap (Multi-domain but No Adaptation)
        try:
            vis_s1_t, vis_s1_l, vis_s1_d, vis_s1_la = load_imdb("test", max_samples=150)
            vis_s2_t, vis_s2_l, vis_s2_d, vis_s2_la = load_yelp("test", max_samples=150)
            vis_src_t, vis_src_l, vis_src_d, vis_src_la = vis_s1_t + vis_s2_t, vis_s1_l + vis_s2_l, vis_s1_d + vis_s2_d, vis_s1_la + vis_s2_la
            
            vis_tgt_t, vis_tgt_l, vis_tgt_d, vis_tgt_la = load_amazon_split("english", "all", "test", max_samples=300)
            ld_src = make_dataloader(vis_src_t, vis_src_l, vis_src_d, vis_src_la, tokenizer, batch_size=BATCH_SIZE)
            ld_tgt = make_dataloader(vis_tgt_t, vis_tgt_l, vis_tgt_d, vis_tgt_la, tokenizer, batch_size=BATCH_SIZE)
            visualize_tsne(model, tokenizer, [ld_src, ld_tgt], ["Sources (IMDb+Yelp)", "Target (Amazon)"], device, "S5_MultiSource_Gap_Before_DANN")
        except Exception as e:
            print(f"⚠️ Không thể tạo biểu đồ t-SNE: {e}")

    # --- S6a: Single-Source Domain Adaptation DANN (Source: IMDb, Target: Amazon) ---
    if args.s in ["0", "6", "6a"]:
        print_banner("Scenario 6a: Single-Source Domain Adaptation DANN (Source: IMDb, Target: Amazon)")
        s_texts, s_labels, s_d_ids, s_la_ids = load_imdb("train", max_samples=MAX_TRAIN)
        # Target data (Amazon) MUST BE unlabeled for Unsupervised Domain Adaptation
        t_texts, t_labels, t_d_ids, t_la_ids = load_amazon_split("english", "all", "train", max_samples=MAX_TRAIN, unlabeled=True)
        
        s_train, s_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(s_texts, s_labels, s_d_ids, s_la_ids, test_size=0.1, random_state=42)
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
                print("🚀 Nạp não bộ đã học từ IMDb vào DANN để tránh sụp đổ đối nghịch...")
                model_dann.load_state_dict(torch.load("checkpoints/model_imdb.pt", map_location=device), strict=False)
            
            weights = compute_class_weights(l_train)
            model_dann = train_dann(model_dann, tokenizer, s_loader, t_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR/10.0, device=device, class_weights=weights)
            torch.save(model_dann.state_dict(), checkpoint_path)
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s6a = evaluate_model(model_dann, test_loader, device, "S6a_DANN_Amazon")
        save_results(res_s6a, "results/results_s6a.json")
        
        # Visualize Domain Alignment (After DANN)
        try:
            vis_src_t, vis_src_l, vis_src_d, vis_src_la = load_imdb("test", max_samples=300)
            vis_tgt_t, vis_tgt_l, vis_tgt_d, vis_tgt_la = load_amazon_split("english", "all", "test", max_samples=300)
            ld_src = make_dataloader(vis_src_t, vis_src_l, vis_src_d, vis_src_la, tokenizer, batch_size=BATCH_SIZE)
            ld_tgt = make_dataloader(vis_tgt_t, vis_tgt_l, vis_tgt_d, vis_tgt_la, tokenizer, batch_size=BATCH_SIZE)
            visualize_tsne(model_dann, tokenizer, [ld_src, ld_tgt], ["Source (IMDb)", "Target (Amazon)"], device, "S6a_Domain_Alignment_After_DANN")
        except Exception as e:
            print(f"⚠️ Không thể tạo biểu đồ t-SNE: {e}")

    # --- S6b: Multi-Source Domain Adaptation DANN (Source: IMDb + Yelp, Target: Amazon) ---
    if args.s in ["0", "6", "6b"]:
        print_banner("Scenario 6b: Multi-Source Domain Adaptation DANN (Source: IMDb + Yelp, Target: Amazon)")
        t1, l1, d1, la1 = load_imdb("train", max_samples=MAX_TRAIN//2)
        t2, l2, d2, la2 = load_yelp("train", max_samples=MAX_TRAIN//2)
        s_texts, s_labels, s_d_ids, s_la_ids = t1 + t2, l1 + l2, d1 + d2, la1 + la2
        # Target data (Amazon) MUST BE unlabeled for Unsupervised Domain Adaptation
        t_texts, t_labels, t_d_ids, t_la_ids = load_amazon_split("english", "all", "train", max_samples=MAX_TRAIN, unlabeled=True)
        
        s_train, s_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(s_texts, s_labels, s_d_ids, s_la_ids, test_size=0.1, random_state=42)
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
                print("🚀 Nạp não bộ Multi-domain (IMDb+Yelp) vào DANN...")
                model_dann.load_state_dict(torch.load("checkpoints/model_s5_multidomain.pt", map_location=device), strict=False)
            
            weights = compute_class_weights(l_train)
            model_dann = train_dann(model_dann, tokenizer, s_loader, t_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR/10.0, device=device, class_weights=weights)
            torch.save(model_dann.state_dict(), checkpoint_path)
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s6b = evaluate_model(model_dann, test_loader, device, "S6b_MultiSource_DANN_Amazon")
        save_results(res_s6b, "results/results_s6b.json")
        
        # Visualize Domain Alignment (After Multi-Source DANN)
        try:
            vis_s1_t, vis_s1_l, vis_s1_d, vis_s1_la = load_imdb("test", max_samples=150)
            vis_s2_t, vis_s2_l, vis_s2_d, vis_s2_la = load_yelp("test", max_samples=150)
            vis_src_t, vis_src_l, vis_src_d, vis_src_la = vis_s1_t + vis_s2_t, vis_s1_l + vis_s2_l, vis_s1_d + vis_s2_d, vis_s1_la + vis_s2_la
            
            vis_tgt_t, vis_tgt_l, vis_tgt_d, vis_tgt_la = load_amazon_split("english", "all", "test", max_samples=300)
            ld_src = make_dataloader(vis_src_t, vis_src_l, vis_src_d, vis_src_la, tokenizer, batch_size=BATCH_SIZE)
            ld_tgt = make_dataloader(vis_tgt_t, vis_tgt_l, vis_tgt_d, vis_tgt_la, tokenizer, batch_size=BATCH_SIZE)
            visualize_tsne(model_dann, tokenizer, [ld_src, ld_tgt], ["Sources (IMDb+Yelp)", "Target (Amazon)"], device, "S6b_MultiSource_Domain_Alignment")
        except Exception as e:
            print(f"⚠️ Không thể tạo biểu đồ t-SNE: {e}")

    # --- S7: Supervised Target Upper Bound (Amazon -> Amazon) ---
    if args.s in ["0", "7"]:
        print_banner("Scenario 7: Supervised Target Upper Bound (Amazon -> Amazon)")
        t_all, l_all, d_all, la_all = load_amazon_split("english", "all", "train", max_samples=MAX_TRAIN)
        t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.1, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        
        model = BaseModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_upper_bound_s7.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            weights = compute_class_weights(l_train)
            model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
            torch.save(model.state_dict(), checkpoint_path)
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s7 = evaluate_model(model, test_loader, device, "S7_UpperBound_Amazon")
        save_results(res_s7, "results/results_s7.json")

    # --- S8: Multi-domain Evaluation (Amz+IMDb+Yelp -> Each) ---
    if args.s in ["0", "8"]:
        print_banner("Scenario 8: Multi-domain Evaluation (Amz+IMDb+Yelp)")
        t1, l1, d1, la1 = load_amazon_split("english", "all", "train", max_samples=MAX_TRAIN//3)
        t2, l2, d2, la2 = load_imdb("train", max_samples=MAX_TRAIN//3)
        t3, l3, d3, la3 = load_yelp("train", max_samples=MAX_TRAIN//3)
        
        t_all, l_all, d_all, la_all = t1+t2+t3, l1+l2+l3, d1+d2+d3, la1+la2+la3
        t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.1, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        
        model = BaseModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_s8_mdl.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            weights = compute_class_weights(l_train)
            model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
            torch.save(model.state_dict(), checkpoint_path)
        
        for domain_name, loader_func in [("Amazon", load_amazon_split), ("IMDb", load_imdb), ("Yelp", load_yelp)]:
            print(f"\n--- S8: Testing on {domain_name} ---")
            if domain_name == "Amazon":
                tt, tl, td, tla = loader_func("english", "all", "test", max_samples=MAX_TEST)
            else:
                tt, tl, td, tla = loader_func("test", max_samples=MAX_TEST)
            
            t_loader = make_dataloader(tt, tl, td, tla, tokenizer, batch_size=BATCH_SIZE)
            res = evaluate_model(model, t_loader, device, f"S8_MDL_{domain_name}")
            save_results(res, f"results/results_s8_{domain_name.lower()}.json")

    # --- S9: Supervised Fine-tuning (SFT) ---
    if args.s in ["0", "9", "9a", "9b"]:
        # S9a: Single-Source (IMDb) -> Amazon SFT
        if args.s in ["0", "9", "9a"]:
            print_banner("Scenario 9a: Single-Source SFT (IMDb -> 500 Amazon)")
            model = BaseModel(config["model"]["name"])
            if os.path.exists("checkpoints/model_imdb.pt"):
                model.load_state_dict(torch.load("checkpoints/model_imdb.pt", map_location=device))
            else:
                print("⚠️ Cần chạy S1a trước.")
            
            # Load 200 labeled Amazon samples
            t_amz, l_amz, d_amz, la_amz = load_amazon_split("english", "all", "train", max_samples=500)
            train_loader = make_dataloader(t_amz, l_amz, d_amz, la_amz, tokenizer, batch_size=8, shuffle=True)
            
            checkpoint_path = "checkpoints/model_sft_s9a.pt"
            if os.path.exists(checkpoint_path):
                print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            else:
                # Fine-tune with very low LR
                model = train_model(model, tokenizer, train_loader, num_epochs=3, lr=5e-6, device=device)
                torch.save(model.state_dict(), checkpoint_path)
            
            tt, tl, td, tla = load_amazon_split("english", "all", "test", max_samples=MAX_TEST)
            test_loader = make_dataloader(tt, tl, td, tla, tokenizer, batch_size=BATCH_SIZE)
            res_s9a = evaluate_model(model, test_loader, device, "S9a_SFT_IMDb_Amazon")
            save_results(res_s9a, "results/results_s9a.json")

        # S9b: Multi-Source (IMDb + Yelp) -> Amazon SFT
        if args.s in ["0", "9", "9b"]:
            print_banner("Scenario 9b: Multi-Source SFT (IMDb+Yelp -> 500 Amazon)")
            model = BaseModel(config["model"]["name"])
            if os.path.exists("checkpoints/model_s5_multidomain.pt"):
                model.load_state_dict(torch.load("checkpoints/model_s5_multidomain.pt", map_location=device))
            else:
                print("⚠️ Cần chạy S5 trước.")
            
            # Load same 500 labeled Amazon samples
            t_amz, l_amz, d_amz, la_amz = load_amazon_split("english", "all", "train", max_samples=500)
            train_loader = make_dataloader(t_amz, l_amz, d_amz, la_amz, tokenizer, batch_size=8, shuffle=True)
            
            checkpoint_path = "checkpoints/model_sft_s9b.pt"
            if os.path.exists(checkpoint_path):
                print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            else:
                model = train_model(model, tokenizer, train_loader, num_epochs=3, lr=5e-6, device=device)
                torch.save(model.state_dict(), checkpoint_path)
            
            tt, tl, td, tla = load_amazon_split("english", "all", "test", max_samples=MAX_TEST)
            test_loader = make_dataloader(tt, tl, td, tla, tokenizer, batch_size=BATCH_SIZE)
            res_s9b = evaluate_model(model, test_loader, device, "S9b_SFT_Multi_Amazon")
            save_results(res_s9b, "results/results_s9b.json")

    # --- S10a: Multi-source Cross-lingual Transfer (IMDb + Yelp -> VSFC) ---
    if args.s in ["0", "10", "10a"]:
        print_banner("Scenario 10a: Multi-source Cross-lingual Transfer (IMDb+Yelp -> VSFC)")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_s5_multidomain.pt"):
            model.load_state_dict(torch.load("checkpoints/model_s5_multidomain.pt", map_location=device))
        else:
            print("⚠️ Cần chạy S5 trước.")
        
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=MAX_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        res_s10a = evaluate_model(model, test_loader_vi, device, "S10a_MultiCross_ZeroShot")
        save_results(res_s10a, "results/results_s10a.json")
        
        # Visualize Cross-lingual Alignment (Before DANN)
        try:
            vis_en_t, vis_en_l, vis_en_d, vis_en_la = load_imdb("test", max_samples=300)
            vis_vi_t, vis_vi_l, vis_vi_d, vis_vi_la = load_vsfc("test", max_samples=300)
            ld_en = make_dataloader(vis_en_t, vis_en_l, vis_en_d, vis_en_la, tokenizer, batch_size=BATCH_SIZE)
            ld_vi = make_dataloader(vis_vi_t, vis_vi_l, vis_vi_d, vis_vi_la, tokenizer, batch_size=BATCH_SIZE)
            visualize_tsne(model, tokenizer, [ld_en, ld_vi], ["English (IMDb)", "Vietnamese (VSFC)"], device, "S10a_MultiCross_Alignment_Before")
        except Exception as e:
            print(f"⚠️ Không thể tạo biểu đồ t-SNE: {e}")

    # --- S10b: Multi-source Cross-lingual DANN (IMDb + Yelp -> VSFC) ---
    if args.s in ["0", "10", "10b"]:
        print_banner("Scenario 10b: Multi-source Cross-lingual DANN (IMDb+Yelp -> VSFC)")
        t1, l1, d1, la1 = load_imdb("train", max_samples=MAX_TRAIN//2)
        t2, l2, d2, la2 = load_yelp("train", max_samples=MAX_TRAIN//2)
        s_texts, s_labels, s_d_ids, s_la_ids = t1 + t2, l1 + l2, d1 + d2, la1 + la2
        # Target data (VSFC) unlabeled
        t_texts, t_labels, t_d_ids, t_la_ids = load_vsfc("train", max_samples=MAX_TRAIN, unlabeled=True)
        
        s_train, s_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(s_texts, s_labels, s_d_ids, s_la_ids, test_size=0.1, random_state=42)
        s_loader = make_dataloader(s_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(s_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        t_loader = make_dataloader(t_texts, t_labels, t_d_ids, t_la_ids, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        
        model_dann = DANNModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_dann_s10b.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model_dann.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            if os.path.exists("checkpoints/model_s5_multidomain.pt"):
                model_dann.load_state_dict(torch.load("checkpoints/model_s5_multidomain.pt", map_location=device), strict=False)
            
            weights = compute_class_weights(l_train)
            model_dann = train_dann(model_dann, tokenizer, s_loader, t_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR/10.0, device=device, class_weights=weights)
            torch.save(model_dann.state_dict(), checkpoint_path)
        
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=MAX_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        res_s10b = evaluate_model(model_dann, test_loader_vi, device, "S10b_MultiCross_DANN")
        save_results(res_s10b, "results/results_s10b.json")
        
        # Visualize Cross-lingual Alignment (After DANN)
        try:
            vis_en_t, vis_en_l, vis_en_d, vis_en_la = load_imdb("test", max_samples=300)
            vis_vi_t, vis_vi_l, vis_vi_d, vis_vi_la = load_vsfc("test", max_samples=300)
            ld_en = make_dataloader(vis_en_t, vis_en_l, vis_en_d, vis_en_la, tokenizer, batch_size=BATCH_SIZE)
            ld_vi = make_dataloader(vis_vi_t, vis_vi_l, vis_vi_d, vis_vi_la, tokenizer, batch_size=BATCH_SIZE)
            visualize_tsne(model_dann, tokenizer, [ld_en, ld_vi], ["English (IMDb)", "Vietnamese (VSFC)"], device, "S10b_MultiCross_Alignment_After")
        except Exception as e:
            print(f"⚠️ Không thể tạo biểu đồ t-SNE: {e}")

    print_banner("ALL EXPERIMENTS COMPLETED")
    try:
        generate_aggregate_report()
    except Exception as e:
        print(f"⚠️ Không thể tạo báo cáo tổng hợp: {e}")

if __name__ == "__main__":
    main()
