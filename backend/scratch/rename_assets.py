import os
import shutil

results_dir = r"d:\XuanDuc\TaiLieuKi8\CuoiKiCd4\project\backend\results"
checkpoints_dir = r"d:\XuanDuc\TaiLieuKi8\CuoiKiCd4\project\backend\checkpoints"

RENAME_MAP = {
    "results_s0.json": "results_s6.json",
    "results_s1.json": "results_s2.json",
    "results_s1a.json": "results_s1.json",
    "results_s1b.json": "results_s3.json",
    "results_s2.json": "results_s4.json",
    "results_s3.json": "results_s5.json",
    "results_s4.json": "results_s7.json",
    "results_s4b.json": "results_s8.json",
    "results_s5.json": "results_s9.json",
    "results_s6.json": "results_s10.json",
    "results_s7.json": "results_s11.json",
    "results_s8.json": "results_s12.json",
    "results_s9.json": "results_s13.json",
    "results_s10.json": "results_s14.json",
    "results_s11.json": "results_s15.json",
    "results_s12.json": "results_s16.json",
}

print("[INFO] Renaming result files...")
for src, dst in RENAME_MAP.items():
    src_path = os.path.join(results_dir, src)
    dst_path = os.path.join(results_dir, dst)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"[OK] Moved {src} -> {dst}")
    else:
        print(f"[WARN] {src} not found, skipping.")

print("\n[INFO] Renaming checkpoint files...")
checkpoint_src = os.path.join(checkpoints_dir, "model_s8_unified_dann.pt")
checkpoint_dst = os.path.join(checkpoints_dir, "model_s12_unified_dann.pt")
if os.path.exists(checkpoint_src):
    shutil.move(checkpoint_src, checkpoint_dst)
    print("[OK] Moved model_s8_unified_dann.pt -> model_s12_unified_dann.pt")
else:
    print("[WARN] model_s8_unified_dann.pt not found, skipping.")

print("\n[INFO] Asset renaming completed!")
