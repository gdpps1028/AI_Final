# from pathlib import Path
# import re

# # Label 檔案路徑
# label_dir = Path("D:/chenm/nycu/AI/AI_Final/yolo/dataset/labels")
# label_paths = list(label_dir.rglob("*.txt"))

# # 讀取所有 class id
# class_ids = set()
# for file in label_paths:
#     lines = file.read_text(encoding="utf-8").splitlines()
#     for line in lines:
#         parts = line.strip().split()
#         if parts and parts[0].isdigit():
#             class_ids.add(int(parts[0]))

# # 排序 class id
# class_ids = sorted(class_ids)
# nc = len(class_ids)

# # 產生 names 陣列（此處直接用數字代替）
# names_list = [str(cid) for cid in class_ids]

# # YAML 內容
# yaml_content = f"""train: D:/chenm/nycu/AI/AI_Final/yolo/dataset/images/train
# val: D:/chenm/nycu/AI/AI_Final/yolo/dataset/images/val
# test: D:/chenm/nycu/AI/AI_Final/yolo/dataset/images/test

# nc: {nc}
# names: [{", ".join(names_list)}]
# """

# # 寫入 data.yaml
# yaml_path = Path("D:/chenm/nycu/AI/AI_Final/yolo/dataset/data.yaml")
# yaml_path.write_text(yaml_content, encoding="utf-8")

# print(f"✅ data.yaml 已建立！共 {nc} 個類別")

from pathlib import Path
import yaml

root_dir = Path("D:/chenm/nycu/AI/AI_Final/yolo")
char2id_path = root_dir / "train_dataset/char2id.txt"
output_yaml = root_dir / "dataset/data.yaml"

# 讀取 char2id 並排序
char2id = {}
with open(char2id_path, encoding="utf-8") as f:
    for line in f:
        ch, idx = line.strip().split('\t')
        char2id[int(idx)] = ch

names = [char2id[i] for i in sorted(char2id)]

data = {
    "path": str(root_dir / "dataset").replace("\\", "/"),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(names),
    "names": names
}

# 儲存為 data.yaml
with open(output_yaml, "w", encoding="utf-8") as f:
    yaml.dump(data, f, allow_unicode=True)

print("✅ 產生完成：", output_yaml)

