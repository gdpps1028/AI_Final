from pathlib import Path
import shutil

label_root = Path("D:/chenm/nycu/AI/AI_Final/yolo/dataset/labels")
image_root = Path("D:/chenm/nycu/AI/AI_Final/yolo/dataset/images")

# 對 train/val/test 三個資料集
for split in ["train", "val", "test"]:
    img_dir = image_root / split
    lbl_dir = label_root / split

    # 掃描所有圖片
    for img_path in img_dir.glob("*.png"):
        char = img_path.stem.split("_")[0]  # 擷取「乙」← 乙_0.png
        src_label = lbl_dir / f"{char}.txt"
        dst_label = lbl_dir / f"{img_path.stem}.txt"
        if src_label.exists():
            shutil.copy(src_label, dst_label)
        else:
            print(f"⚠️ 沒找到 label: {src_label.name}")
