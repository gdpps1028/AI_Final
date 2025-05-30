# from pathlib import Path
# import shutil
# import random
# from collections import defaultdict

# # 設定來源路徑
# src_images = Path("D:/chenm/nycu/AI/AI_Final/yolo/train_dataset/images")
# src_labels = Path("D:/chenm/nycu/AI/AI_Final/yolo/train_dataset/labels")

# # 設定目標根目錄
# dest_root = Path("D:/chenm/nycu/AI/AI_Final/yolo/dataset")

# # 定義三種資料集
# splits = ["train", "val", "test"]

# # 建立 images/labels 各自的 train/val/test 子資料夾
# for split in splits:
#     (dest_root / "images" / split).mkdir(parents=True, exist_ok=True)
#     (dest_root / "labels" / split).mkdir(parents=True, exist_ok=True)

# # 對每個字的所有圖片進行分組
# char_to_images = defaultdict(list)
# for img_path in src_images.glob("*.png"):
#     char = img_path.stem.split("_")[0]
#     char_to_images[char].append(img_path)

# # 對每個字各自切成 8:1:1
# for char, images in char_to_images.items():
#     random.shuffle(images)
#     total = len(images)
#     n_train = int(total * 0.8)
#     n_val = int(total * 0.1)
#     n_test = total - n_train - n_val

#     split_imgs = {
#         "train": images[:n_train],
#         "val": images[n_train:n_train + n_val],
#         "test": images[n_train + n_val:]
#     }

#     for split, img_list in split_imgs.items():
#         for img_path in img_list:
#             # 複製圖片
#             shutil.copy(img_path, dest_root / "images" / split / img_path.name)

#             # 對應的 label（如 "乙.txt"）
#             label_path = src_labels / f"{char}.txt"
#             if label_path.exists():
#                 shutil.copy(label_path, dest_root / "labels" / split / label_path.name)
#             else:
#                 print(f"⚠️ 沒找到對應標籤: {label_path.name}")
from pathlib import Path
import shutil
import random
from collections import defaultdict

# 設定來源路徑
src_images = Path("D:/chenm/nycu/AI/AI_Final/yolo/train_dataset/images")
src_labels = Path("D:/chenm/nycu/AI/AI_Final/yolo/train_dataset/labels")  # 修正這裡: 單字 label 來源

# 設定目標根目錄
dest_root = Path("D:/chenm/nycu/AI/AI_Final/yolo/dataset")

# 定義三種資料集
splits = ["train", "val", "test"]

# 建立 images/labels 各自的 train/val/test 子資料夾
for split in splits:
    (dest_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (dest_root / "labels" / split).mkdir(parents=True, exist_ok=True)

# 對每個字的所有圖片進行分組
char_to_images = defaultdict(list)
for img_path in src_images.glob("*.png"):
    char = img_path.stem.split("_")[0]
    char_to_images[char].append(img_path)

# 對每個字各自切成 8:1:1 並複製圖片與 label
for char, images in char_to_images.items():
    random.shuffle(images)
    total = len(images)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    n_test = total - n_train - n_val

    split_imgs = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, img_list in split_imgs.items():
        for img_path in img_list:
            # 複製圖片
            dest_img_path = dest_root / "images" / split / img_path.name
            shutil.copy(img_path, dest_img_path)

            # 複製對應的 label（乙.txt → 乙_0.txt）
            label_src_path = src_labels / f"{char}.txt"
            label_dest_path = dest_root / "labels" / split / f"{img_path.stem}.txt"
            if label_src_path.exists():
                shutil.copy(label_src_path, label_dest_path)
            else:
                print(f"⚠️ 沒找到對應標籤: {label_src_path.name}")
