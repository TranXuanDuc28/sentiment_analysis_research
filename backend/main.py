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
from src.model import BaseModel, DANNModel, UnifiedFrameworkModel
from src.train import train_model, train_dann, train_multitask, compute_class_weights
from src.evaluate import evaluate_model
from src.visualize_embeddings import visualize_tsne
from src.report_generator import generate_aggregate_report
from src.utils import print_banner, save_results, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=str, default="0", help="Scenario to run (0=all, 1, 2, 3, 4, 5, 6, 7)")
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
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    # =========================================================================
    # PHASE 1: MULTIDOMAIN ANALYSIS (RQ1)
    # =========================================================================

    # --- S1: Transfer Learning (IMDb -> Amazon) ---
    if args.s in ["0", "1"] and config["scenarios"].get("run_s1", True):
        print_banner("Scenario 1: Transfer Learning (IMDb -> Amazon)")
        # 1. Train on IMDb (Source)
        t_all, l_all, d_all, la_all = load_imdb("train", max_samples=BASE_TRAIN)
        t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.2, random_state=42)
        train_loader = make_dataloader(t_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, la_val, tokenizer, batch_size=BATCH_SIZE)
        
        model = BaseModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_s1_imdb.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            weights = compute_class_weights(l_train)
            model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
            torch.save(model.state_dict(), checkpoint_path)
        
        # 2. Evaluate on Amazon (Target)
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s1 = evaluate_model(model, test_loader, device, "S1_Transfer_IMDb_Amazon")
        save_results(res_s1, "results/results_s1.json")

    # --- S2: Domain Adaptation (DANN: IMDb + Yelp -> Amazon) ---
    if args.s in ["0", "2"] and config["scenarios"].get("run_s2", True):
        print_banner("Scenario 2: Domain Adaptation (IMDb+Yelp -> Amazon)")
        t1, l1, d1, la1 = load_imdb("train", max_samples=BASE_TRAIN)
        t2, l2, d2, la2 = load_yelp("train", max_samples=BASE_TRAIN)
        s_texts, s_labels, s_d_ids, s_la_ids = t1 + t2, l1 + l2, d1 + d2, la1 + la2
        t_texts, t_labels, t_d_ids, t_la_ids = load_amazon_split("english", "all", "train", max_samples=BASE_TRAIN * 2, unlabeled=True)
        
        s_train, s_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(s_texts, s_labels, s_d_ids, s_la_ids, test_size=0.2, random_state=42)
        s_loader = make_dataloader(s_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        t_loader = make_dataloader(t_texts, t_labels, t_d_ids, t_la_ids, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        
        model_dann = DANNModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_s2_dann.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model_dann.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            weights = compute_class_weights(l_train)
            model_dann = train_dann(model_dann, tokenizer, s_loader, t_loader, num_epochs=EPOCHS, lr=LR/10.0, device=device, class_weights=weights)
            torch.save(model_dann.state_dict(), checkpoint_path)
        
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s2 = evaluate_model(model_dann, test_loader, device, "S2_DANN_Amazon")
        save_results(res_s2, "results/results_s2.json")

    # --- S3: Multi-task Learning (Sentiment + Domain) ---
    if args.s in ["0", "3"] and config["scenarios"].get("run_s3", True):
        print_banner("Scenario 3: Multi-task Learning (Sentiment + Domain)")
        # Joint training on IMDb + Yelp + Amazon (Source domains)
        t1, l1, d1, la1 = load_imdb("train", max_samples=BASE_TRAIN)
        t2, l2, d2, la2 = load_yelp("train", max_samples=BASE_TRAIN)
        t3, l3, d3, la3 = load_amazon_split("english", "all", "train", max_samples=BASE_TRAIN)
        t_all, l_all, d_all, la_all = t1+t2+t3, l1+l2+l3, d1+d2+d3, la1+la2+la3
        
        t_train, t_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.2, random_state=42)
        train_loader = make_dataloader(t_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        
        model_mtl = UnifiedFrameworkModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_s3_mtl.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model_mtl.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            weights = compute_class_weights(l_train)
            model_mtl = train_multitask(model_mtl, tokenizer, train_loader, num_epochs=EPOCHS, lr=LR, device=device, class_weights=weights)
            torch.save(model_mtl.state_dict(), checkpoint_path)
            
        test_texts, test_labels, test_d_ids, test_la_ids = load_amazon_split("english", "all", "test", max_samples=BASE_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, test_la_ids, tokenizer, batch_size=BATCH_SIZE)
        res_s3 = evaluate_model(model_mtl, test_loader, device, "S3_MTL_Amazon")
        save_results(res_s3, "results/results_s3.json")


    # =========================================================================
    # PHASE 2: MULTILINGUAL ANALYSIS (RQ2)
    # =========================================================================

    # --- S4: Zero-shot Cross-lingual (IMDb EN -> VSFC VI) ---
    if args.s in ["0", "4"] and config["scenarios"].get("run_s4", True):
        print_banner("Scenario 4: Zero-shot Cross-lingual (IMDb -> VSFC)")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_s1_imdb.pt"):
            model.load_state_dict(torch.load("checkpoints/model_s1_imdb.pt", map_location=device))
        
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        res_s4 = evaluate_model(model, test_loader_vi, device, "S4_ZeroShot_VSFC")
        save_results(res_s4, "results/results_s4.json")

    # --- S5: Few-shot Fine-tuning (VI) ---
    if args.s in ["0", "5"] and config["scenarios"].get("run_s5", True):
        print_banner(f"Scenario 5: Few-shot Fine-tuning ({FEW_SHOT} VSFC)")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_s1_imdb.pt"):
            model.load_state_dict(torch.load("checkpoints/model_s1_imdb.pt", map_location=device))
        
        t_vsfc, l_vsfc, d_vsfc, la_vsfc = load_vsfc("train", max_samples=FEW_SHOT)
        train_loader = make_dataloader(t_vsfc, l_vsfc, d_vsfc, la_vsfc, tokenizer, batch_size=8, shuffle=True)
        
        checkpoint_path = "checkpoints/model_s5_fewshot.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            model = train_model(model, tokenizer, train_loader, num_epochs=3, lr=5e-6, device=device)
            torch.save(model.state_dict(), checkpoint_path)
        
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        res_s5 = evaluate_model(model, test_loader_vi, device, "S5_FewShot_VSFC")
        save_results(res_s5, "results/results_s5.json")

    # --- S6: Translation-Based Baseline (VSFC -> Translated EN) ---
    if args.s in ["0", "6"] and config["scenarios"].get("run_s6", True):
        print_banner("Scenario 6: Translation-Based Baseline (VSFC -> EN)")
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_s1_imdb.pt"):
            model.load_state_dict(torch.load("checkpoints/model_s1_imdb.pt", map_location=device))
            
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
        
        print("🌍 Translating test data (Simulated subset)...")
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='vi', target='en')
            test_texts_translated = []
            from tqdm import tqdm
            # Subset for speed in demo
            subset_size = min(500, len(test_texts_vi))
            for text in tqdm(test_texts_vi[:subset_size], desc="Translating"):
                try:
                    trans = translator.translate(text[:4999])
                    test_texts_translated.append(trans if trans else text)
                except Exception:
                    test_texts_translated.append(text)
            
            test_loader_trans = make_dataloader(test_texts_translated, test_labels_vi[:subset_size], test_d_ids_vi[:subset_size], test_la_ids_vi[:subset_size], tokenizer, batch_size=BATCH_SIZE)
            res_s6 = evaluate_model(model, test_loader_trans, device, "S6_Translation_VSFC")
            save_results(res_s6, "results/results_s6.json")
        except ImportError:
            print("⚠️ deep-translator not installed, skipping translation.")


    # =========================================================================
    # PHASE 3: COMBINED FRAMEWORK (RQ3)
    # =========================================================================

    # --- S7: Unified S+D+L Framework ---
    if args.s in ["0", "7"] and config["scenarios"].get("run_s7", True):
        print_banner("Scenario 7: Unified S+D+L Framework")
        # All source domains + unlabeled Vietnamese target
        t1, l1, d1, la1 = load_imdb("train", max_samples=BASE_TRAIN)
        t2, l2, d2, la2 = load_yelp("train", max_samples=BASE_TRAIN)
        t3, l3, d3, la3 = load_amazon_split("english", "all", "train", max_samples=BASE_TRAIN)
        t_vi, l_vi, d_vi, la_vi = load_vsfc("train", max_samples=BASE_TRAIN, unlabeled=True)
        
        t_all = t1 + t2 + t3 + t_vi
        l_all = l1 + l2 + l3 + l_vi
        d_all = d1 + d2 + d3 + d_vi
        la_all = la1 + la2 + la3 + la_vi
        
        s_train, s_val, l_train, l_val, d_train, d_val, la_train, la_val = train_test_split(t_all, l_all, d_all, la_all, test_size=0.1, random_state=42)
        train_loader = make_dataloader(s_train, l_train, d_train, la_train, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        
        model_unified = UnifiedFrameworkModel(config["model"]["name"])
        checkpoint_path = "checkpoints/model_s7_unified.pt"
        if os.path.exists(checkpoint_path):
            print(f"🚀 Found checkpoint {checkpoint_path}, loading...")
            model_unified.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            weights = compute_class_weights([lbl for lbl in l_train if lbl >= 0])
            model_unified = train_multitask(model_unified, tokenizer, train_loader, num_epochs=EPOCHS, lr=LR/10.0, device=device, class_weights=weights)
            torch.save(model_unified.state_dict(), checkpoint_path)
            
        test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi = load_vsfc("test", max_samples=BASE_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, test_la_ids_vi, tokenizer, batch_size=BATCH_SIZE)
        res_s7 = evaluate_model(model_unified, test_loader_vi, device, "S7_Unified_VSFC")
        save_results(res_s7, "results/results_s7.json")


    # =========================================================================
    # MODEL COMPARISON (XLM-R vs mBERT)
    # =========================================================================
    if config["scenarios"].get("run_comparison", True):
        print_banner("MODEL COMPARISON: XLM-R vs mBERT")
        mbert_name = config["model"]["mbert"]
        tokenizer_mbert = AutoTokenizer.from_pretrained(mbert_name)
        
        # 1. mBERT S4 (Zero-shot)
        print_banner("mBERT Scenario 4 (Zero-shot)")
        model_mb = BaseModel(mbert_name)
        checkpoint_mb = "checkpoints/model_mbert_s1.pt"
        if not os.path.exists(checkpoint_mb):
            t_all, l_all, d_all, la_all = load_imdb("train", max_samples=BASE_TRAIN)
            train_loader = make_dataloader(t_all, l_all, d_all, la_all, tokenizer_mbert, batch_size=BATCH_SIZE, shuffle=True)
            model_mb = train_model(model_mb, tokenizer_mbert, train_loader, num_epochs=EPOCHS, lr=LR, device=device)
            torch.save(model_mb.state_dict(), checkpoint_mb)
        else:
            print(f"🚀 Found checkpoint {checkpoint_mb}, loading...")
            model_mb.load_state_dict(torch.load(checkpoint_mb, map_location=device))
            
        test_vi_t, test_vi_l, test_vi_d, test_vi_la = load_vsfc("test", max_samples=BASE_TEST)
        test_loader_vi = make_dataloader(test_vi_t, test_vi_l, test_vi_d, test_vi_la, tokenizer_mbert, batch_size=BATCH_SIZE)
        res_mb_s4 = evaluate_model(model_mb, test_loader_vi, device, "mBERT_S4_ZeroShot")
        save_results(res_mb_s4, "results/results_mbert_s4.json")

    print_banner("ALL EXPERIMENTS COMPLETED")
    generate_aggregate_report()

if __name__ == "__main__":
    main()
