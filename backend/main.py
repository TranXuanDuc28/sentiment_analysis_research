
import argparse
import yaml
import os
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from src.dataset import load_amazon_split, load_vsfc, load_tweeteval, make_dataloader, word_segment_vietnamese
from src.model import BaseModel, DANNModel
from src.train import train_model, train_dann, compute_class_weights
from src.evaluate import evaluate_model
from src.visualize_embeddings import visualize_tsne
from src.report_generator import generate_aggregate_report
from src.utils import print_banner, save_results, print_dataset_statistics, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=0, help="Scenario to run (0=all, 1-5)")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    MAX_TEST = max(config["scenarios"]["max_samples_test"], 1000)
    
    # Để lưu kết quả đối chứng
    global_results = {}

    # 1. SCENARIO 1: Monolingual Baselines
    if args.s in [0, 1]:
        print_banner("Scenario 1a: Baseline English (XLM-R)")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        t_all, l_all, d_all = load_amazon_split("english", "books", "train", max_samples=config["scenarios"]["max_samples_train"])
        t_train, t_val, l_train, l_val, d_train, d_val = train_test_split(t_all, l_all, d_all, test_size=0.1, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, tokenizer, batch_size=config["training"]["batch_size"])
        
        model = BaseModel(config["model"]["name"])
        weights = compute_class_weights(l_train)
        model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=int(config["training"]["epochs"]), lr=float(config["training"]["learning_rate"]), device=device, class_weights=weights)
        torch.save(model.state_dict(), "checkpoints/model_en_books.pt")
        
        test_texts, test_labels, test_d_ids = load_amazon_split("english", "books", "test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, tokenizer, batch_size=config["training"]["batch_size"])
        global_results["S1a"] = evaluate_model(model, test_loader, device, "S1a_Baseline_EN")

        print_banner("Scenario 1b: Baseline Vietnamese (XLM-R)")
        t_all_vi, l_all_vi, d_all_vi = load_vsfc("train", max_samples=config["scenarios"]["max_samples_train"])
        tv_train, tv_val, lv_train, lv_val, dv_train, dv_val = train_test_split(t_all_vi, l_all_vi, d_all_vi, test_size=0.1, random_state=42)
        
        train_loader_vi = make_dataloader(tv_train, lv_train, dv_train, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader_vi = make_dataloader(tv_val, lv_val, dv_val, tokenizer, batch_size=config["training"]["batch_size"])
        
        model_vi = BaseModel(config["model"]["name"])
        weights_vi = compute_class_weights(lv_train)
        model_vi = train_model(model_vi, tokenizer, train_loader_vi, val_loader=val_loader_vi, num_epochs=int(config["training"]["epochs"]), device=device, class_weights=weights_vi)
        
        test_texts_vi, test_labels_vi, test_d_ids_vi = load_vsfc("test", max_samples=MAX_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, tokenizer, batch_size=config["training"]["batch_size"])
        global_results["S1b"] = evaluate_model(model_vi, test_loader_vi, device, "S1b_XLMR_VI")
        
        print("\nComparing with PhoBERT baseline...")
        try:
            phobert_tok = AutoTokenizer.from_pretrained(config["model"]["vi_baseline"])
            phobert_model = BaseModel(config["model"]["vi_baseline"])
            tv_train_seg = word_segment_vietnamese(tv_train)
            tv_val_seg = word_segment_vietnamese(tv_val)
            train_loader_ph = make_dataloader(tv_train_seg, lv_train, dv_train, phobert_tok, batch_size=config["training"]["batch_size"], shuffle=True)
            val_loader_ph = make_dataloader(tv_val_seg, lv_val, dv_val, phobert_tok, batch_size=config["training"]["batch_size"])
            phobert_model = train_model(phobert_model, phobert_tok, train_loader_ph, val_loader=val_loader_ph, num_epochs=int(config["training"]["epochs"]), device=device, class_weights=weights_vi)
            test_texts_vi_seg = word_segment_vietnamese(test_texts_vi)
            test_loader_ph = make_dataloader(test_texts_vi_seg, test_labels_vi, test_d_ids_vi, phobert_tok, batch_size=config["training"]["batch_size"])
            global_results["S1b_PhoBERT"] = evaluate_model(phobert_model, test_loader_ph, device, "S1b_PhoBERT_VI")
        except: pass

    # 2. SCENARIO 2: Zero-shot Cross-lingual (EN -> VI)
    if args.s in [0, 2]:
        print_banner("Scenario 2: Zero-shot Cross-lingual (English -> Vietnamese)")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        model = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_en_books.pt"):
            model.load_state_dict(torch.load("checkpoints/model_en_books.pt", map_location=device))
        
        test_texts_vi, test_labels_vi, test_d_ids_vi = load_vsfc("test", max_samples=MAX_TEST)
        test_loader_vi = make_dataloader(test_texts_vi, test_labels_vi, test_d_ids_vi, tokenizer, batch_size=config["training"]["batch_size"])
        res_s2 = evaluate_model(model, test_loader_vi, device, "S2_ZeroShot_EN_VI")
        
        if "S1b" in global_results:
            gap = global_results["S1b"]["f1_macro"] - res_s2["f1_macro"]
            print(f"\n📊 INSIGHT: Mất {gap*100:.2f}% F1-Macro khi chạy Zero-shot thay vì train trực tiếp.")

    # 3. SCENARIO 3: Unseen Domain Transfer (Books+Electronics -> Apparel)
    if args.s in [0, 3]:
        print_banner("Scenario 3: Unseen Domain Transfer (Books+Electronics -> Apparel)")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        t1, l1, d1 = load_amazon_split("english", "books", "train", max_samples=config["scenarios"]["max_samples_train"]//2)
        t2, l2, d2 = load_amazon_split("english", "electronics", "train", max_samples=config["scenarios"]["max_samples_train"]//2)
        t_all, l_all, d_all = t1 + t2, l1 + l2, d1 + d2
        t_train, t_val, l_train, l_val, d_train, d_val = train_test_split(t_all, l_all, d_all, test_size=0.1, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, tokenizer, batch_size=config["training"]["batch_size"])
        model = BaseModel(config["model"]["name"])
        weights = compute_class_weights(l_train)
        model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=int(config["training"]["epochs"]), device=device, class_weights=weights)
        
        test_texts, test_labels, test_d_ids = load_amazon_split("english", "apparel", "test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, tokenizer, batch_size=config["training"]["batch_size"])
        evaluate_model(model, test_loader, device, "S3_UnseenDomain_Apparel")

    # 4. SCENARIO 4: Domain Adaptation (DANN)
    if args.s in [0, 4]:
        print_banner("Scenario 4a: Adaptation Baseline (No DANN)")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        model_base = BaseModel(config["model"]["name"])
        if os.path.exists("checkpoints/model_en_books.pt"):
            model_base.load_state_dict(torch.load("checkpoints/model_en_books.pt", map_location=device))
        
        test_texts, test_labels, test_d_ids = load_tweeteval("test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, tokenizer, batch_size=config["training"]["batch_size"])
        res_s4a = evaluate_model(model_base, test_loader, device, "S4a_Baseline_NoDANN")
        save_results(res_s4a, "results/results_s4a.json")

        print_banner("Scenario 4b: DANN Adaptation")
        s_texts_all, s_labels_all, s_d_ids_all = load_amazon_split("english", "books", "train", max_samples=config["scenarios"]["max_samples_train"])
        t_texts, t_labels, t_d_ids = load_tweeteval("train", max_samples=config["scenarios"]["max_samples_train"], unlabeled=True)
        s_train, s_val, l_train, l_val, d_train, d_val = train_test_split(s_texts_all, s_labels_all, s_d_ids_all, test_size=0.1, random_state=42)
        s_loader = make_dataloader(s_train, l_train, d_train, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = make_dataloader(s_val, l_val, d_val, tokenizer, batch_size=config["training"]["batch_size"])
        t_loader = make_dataloader(t_texts, t_labels, t_d_ids, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        
        model_dann = DANNModel(config["model"]["name"])
        weights = compute_class_weights(l_train)
        model_dann = train_dann(model_dann, tokenizer, s_loader, t_loader, val_loader=val_loader, num_epochs=int(config["training"]["epochs"]), lr=float(config["training"]["learning_rate"]), device=device, class_weights=weights)
        res_s4b = evaluate_model(model_dann, test_loader, device, "S4b_DANN_Twitter")
        print(f"\n🚀 INSIGHT: DANN giúp cải thiện { (res_s4b['f1_macro'] - res_s4a['f1_macro'])*100:.2f}% F1-Macro trên Twitter.")
        global_results["S4b"] = res_s4b

    # 5. SCENARIO 5: Multilingual Joint Learning (EN + VI)
    if args.s in [0, 5]:
        print_banner("Scenario 5: Multilingual Joint Learning")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        t_en, l_en, d_en = load_amazon_split("english", "books", "train", max_samples=config["scenarios"]["max_samples_train"]//2)
        t_vi, l_vi, d_vi = load_vsfc("train", max_samples=config["scenarios"]["max_samples_train"]//2)
        t_all, l_all, d_all = t_en + t_vi, l_en + l_vi, d_en + d_vi
        t_train, t_val, l_train, l_val, d_train, d_val = train_test_split(t_all, l_all, d_all, test_size=0.1, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, tokenizer, batch_size=config["training"]["batch_size"])
        model = BaseModel(config["model"]["name"])
        weights = compute_class_weights(l_train)
        model = train_model(model, tokenizer, train_loader, val_loader=val_loader, num_epochs=int(config["training"]["epochs"]), device=device, class_weights=weights)
        
        print("\nTesting on Vietnamese...")
        test_vi_t, test_vi_l, test_vi_d = load_vsfc("test", max_samples=MAX_TEST)
        test_loader_vi = make_dataloader(test_vi_t, test_vi_l, test_vi_d, tokenizer, batch_size=config["training"]["batch_size"])
        res_s5_vi = evaluate_model(model, test_loader_vi, device, "S5_Joint_VI")
        
        if "S1b" in global_results:
            gain = res_s5_vi["f1_macro"] - global_results["S1b"]["f1_macro"]
            print(f"\n🚀 INSIGHT: Học đa ngữ giúp tiếng Việt thay đổi {gain*100:.2f}% F1-Macro so với học đơn ngữ.")

    # 6. SCENARIO 6: Hybrid Learning (Bonus)
    if args.s in [0, 6]:
        print_banner("Scenario 6: Hybrid Learning (Multi-source -> Adapt to Twitter)")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        # Mix Books + Electronics as Source
        t1, l1, d1 = load_amazon_split("english", "books", "train", max_samples=config["scenarios"]["max_samples_train"]//2)
        t2, l2, d2 = load_amazon_split("english", "electronics", "train", max_samples=config["scenarios"]["max_samples_train"]//2)
        s_all_t, s_all_l, s_all_d = t1 + t2, l1 + l2, d1 + d2
        
        t_texts, t_labels, t_d_ids = load_tweeteval("train", max_samples=config["scenarios"]["max_samples_train"], unlabeled=True)
        
        s_train, s_val, l_train, l_val, d_train, d_val = train_test_split(s_all_t, s_all_l, s_all_d, test_size=0.1, random_state=42)
        s_loader = make_dataloader(s_train, l_train, d_train, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = make_dataloader(s_val, l_val, d_val, tokenizer, batch_size=config["training"]["batch_size"])
        target_loader = make_dataloader(t_texts, t_labels, t_d_ids, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        
        model_hybrid = DANNModel(config["model"]["name"])
        weights = compute_class_weights(l_train)
        model_hybrid = train_dann(model_hybrid, tokenizer, s_loader, target_loader, val_loader=val_loader, num_epochs=int(config["training"]["epochs"]), lr=float(config["training"]["learning_rate"]), device=device, class_weights=weights)
        
        test_texts, test_labels, test_d_ids = load_tweeteval("test", max_samples=MAX_TEST)
        test_loader = make_dataloader(test_texts, test_labels, test_d_ids, tokenizer, batch_size=config["training"]["batch_size"])
        res_s6 = evaluate_model(model_hybrid, test_loader, device, "S6_Hybrid_Adaptation")
        
        if "S4b" in global_results:
            improvement = res_s6["f1_macro"] - global_results["S4b"]["f1_macro"]
            print(f"\n🚀 INSIGHT: Việc học đa nguồn (Hybrid) giúp Adaptation hiệu quả hơn {improvement*100:.2f}% so với học đơn nguồn (S4b).")

    # 7. SCENARIO 7: The Universal Model (Multilingual + Multidomain)
    if args.s in [0, 7]:
        print_banner("Scenario 7: The Universal Model (EN Books + EN Elec + VI VSFC)")
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        
        # Mix 3 sources equally
        n_per_source = config["scenarios"]["max_samples_train"] // 3
        t1, l1, d1 = load_amazon_split("english", "books", "train", max_samples=n_per_source)
        t2, l2, d2 = load_amazon_split("english", "electronics", "train", max_samples=n_per_source)
        t3, l3, d3 = load_vsfc("train", max_samples=n_per_source)
        
        t_all, l_all, d_all = t1 + t2 + t3, l1 + l2 + l3, d1 + d2 + d3
        t_train, t_val, l_train, l_val, d_train, d_val = train_test_split(t_all, l_all, d_all, test_size=0.1, random_state=42)
        
        train_loader = make_dataloader(t_train, l_train, d_train, tokenizer, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = make_dataloader(t_val, l_val, d_val, tokenizer, batch_size=config["training"]["batch_size"])
        
        model_universal = BaseModel(config["model"]["name"])
        weights = compute_class_weights(l_train)
        model_universal = train_model(model_universal, tokenizer, train_loader, val_loader=val_loader, num_epochs=int(config["training"]["epochs"]), device=device, class_weights=weights)
        
        # Test 1: On Vietnamese
        print("\nTesting Universal Model on Vietnamese...")
        test_vi_t, test_vi_l, test_vi_d = load_vsfc("test", max_samples=MAX_TEST)
        test_loader_vi = make_dataloader(test_vi_t, test_vi_l, test_vi_d, tokenizer, batch_size=config["training"]["batch_size"])
        res_s7_vi = evaluate_model(model_universal, test_loader_vi, device, "S7_Universal_VI")
        
        # Test 2: On Unseen English Domain (Apparel)
        print("\nTesting Universal Model on Unseen English Domain (Apparel)...")
        test_en_t, test_en_l, test_en_d = load_amazon_split("english", "apparel", "test", max_samples=MAX_TEST)
        test_loader_en = make_dataloader(test_en_t, test_en_l, test_en_d, tokenizer, batch_size=config["training"]["batch_size"])
        res_s7_en = evaluate_model(model_universal, test_loader_en, device, "S7_Universal_EN_Unseen")
        
        print("\n📊 FINAL RESEARCH ANALYSIS:")
        try:
            # So sánh với S5 (Joint VI)
            print(f"1. Synergy: So với học đơn miền (S5), việc thêm đa miền giúp tiếng Việt thay đổi { (res_s7_vi['f1_macro'] - res_s5_vi['f1_macro'])*100:.2f}% F1.")
        except: pass
        print(f"2. Cross-lingual Robustness: Mô hình Universal đạt {res_s7_en['f1_macro']*100:.2f}% F1 trên miền chưa từng thấy (Apparel).")

    print_banner("ALL EXPERIMENTS COMPLETED")
    try:
        generate_aggregate_report()
    except Exception as e:
        print(f"⚠️ Không thể tạo báo cáo tổng hợp: {e}")

if __name__ == "__main__":
    main()
