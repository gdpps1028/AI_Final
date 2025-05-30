# import os
# from pathlib import Path

# # 定義每種 IDS 組合對應的 bounding box 結構（x_center, y_center, width, height）
# IDS_BBOX_RULES = {
#     '⿰': [(0.27, 0.5, 0.52, 1.0), (0.76, 0.5, 0.48, 1.0)],
#     '⿱': [(0.51, 0.23, 0.96, 0.42), (0.54, 0.73, 0.92, 0.54)],
#     '⿲': [(0.21, 0.5, 0.38, 1.0), (0.48, 0.5, 0.20, 1.0), (0.77, 0.5, 0.46, 1.0)],
#     '⿳': [(0.51, 0.18, 0.98, 0.34), (0.53, 0.49, 0.94, 0.30), (0.51, 0.80, 0.98, 0.38)],
#     '⿴': [(0.51, 0.5, 1.0, 1.0), (0.51, 0.54, 0.3, 0.32)],
#     '⿵': [(0.51, 0.51, 0.98, 0.96), (0.50, 0.75, 0.24, 0.5)],
#     '⿶': [(0.51, 0.5, 0.98, 0.98), (0.50, 0.34, 0.32, 0.64)],
#     '⿷': [(0.51, 0.5, 0.98, 0.98), (0.72, 0.54, 0.56, 0.4)],
#     '⿸': [(0.51, 0.5, 1.0, 1.0), (0.75, 0.72, 0.5, 0.56)],
#     '⿹': [(0.51, 0.5, 1.0, 1.0), (0.32, 0.68, 0.6, 0.64)],
# }

# IDS_OPERATORS = {op: len(rule) for op, rule in IDS_BBOX_RULES.items()}

# # IDS 結構轉樹狀結構
# def parse_ids(s):
#     stack = []
#     for ch in reversed(s):
#         if ch in IDS_OPERATORS:
#             n = IDS_OPERATORS[ch]
#             children = [stack.pop() for _ in range(n)][::-1]
#             stack.append((ch, children))
#         else:
#             stack.append(ch)
#     return stack[0]

# # 根據 IDS 樹結構套用 bbox
# def assign_bbox(node):
#     if isinstance(node, str):
#         return [(node, 0.5, 0.5, 1.0, 1.0)]
#     op, children = node
#     result = []
#     rules = IDS_BBOX_RULES.get(op)
#     if not rules:
#         # fallback: 均分
#         w = 1.0 / len(children)
#         for i, ch in enumerate(children):
#             result += assign_bbox(ch)
#         return result
#     for ch, (x, y, w, h) in zip(children, rules):
#         if isinstance(ch, str):
#             result.append((ch, x, y, w, h))
#         else:
#             result += assign_bbox(ch)
#     return result

# # 轉為 YOLO 格式
# def to_yolo(class_id, x, y, w, h):
#     return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

# root_dir = Path("D:/chenm/nycu/AI/AI_Final/yolo")
# myids_path = root_dir / "train_dataset/myids.txt"
# output_path = root_dir / "train_dataset/label_txt"
# # 處理 myids.txt
# with open(myids_path, encoding="utf-8") as f:
#     for line in f:
#         parts = line.strip().split("\t")

#         hanzi = parts[2]
#         ids = parts[3]
#         try:
#             tree = parse_ids(ids)
#             bboxes = assign_bbox(tree)

#             with open(output_path / f"{hanzi}.txt", "w", encoding="utf-8") as out:
#                 for ch, x, y, w, h in bboxes:
#                     class_id = ord(ch)
#                     out.write(to_yolo(class_id, x, y, w, h) + "\n")
#         except Exception as e:
#             print(f"[跳過] {hanzi}: {ids} ({e})")


from pathlib import Path

# 定義每種 IDS 組合對應的 bounding box 結構（x_center, y_center, width, height）
IDS_BBOX_RULES = {
    '⿰': [(0.27, 0.5, 0.52, 1.0), (0.76, 0.5, 0.48, 1.0)],
    '⿱': [(0.51, 0.23, 0.96, 0.42), (0.54, 0.73, 0.92, 0.54)],
    '⿲': [(0.21, 0.5, 0.38, 1.0), (0.48, 0.5, 0.20, 1.0), (0.77, 0.5, 0.46, 1.0)],
    '⿳': [(0.51, 0.18, 0.98, 0.34), (0.53, 0.49, 0.94, 0.30), (0.51, 0.80, 0.98, 0.38)],
    '⿴': [(0.51, 0.5, 1.0, 1.0), (0.51, 0.54, 0.3, 0.32)],
    '⿵': [(0.51, 0.51, 0.98, 0.96), (0.50, 0.75, 0.24, 0.5)],
    '⿶': [(0.51, 0.5, 0.98, 0.98), (0.50, 0.34, 0.32, 0.64)],
    '⿷': [(0.51, 0.5, 0.98, 0.98), (0.72, 0.54, 0.56, 0.4)],
    '⿸': [(0.51, 0.5, 1.0, 1.0), (0.75, 0.72, 0.5, 0.56)],
    '⿹': [(0.51, 0.5, 1.0, 1.0), (0.32, 0.68, 0.6, 0.64)],
}

IDS_OPERATORS = {op: len(rule) for op, rule in IDS_BBOX_RULES.items()}

# IDS 結構轉樹狀結構
def parse_ids(s):
    stack = []
    for ch in reversed(s):
        if ch in IDS_OPERATORS:
            n = IDS_OPERATORS[ch]
            children = [stack.pop() for _ in range(n)][::-1]
            stack.append((ch, children))
        else:
            stack.append(ch)
    return stack[0]

# 根據 IDS 樹結構套用 bbox
def assign_bbox(node):
    if isinstance(node, str):
        return [(node, 0.5, 0.5, 1.0, 1.0)]
    op, children = node
    result = []
    rules = IDS_BBOX_RULES.get(op)
    if not rules:
        w = 1.0 / len(children)
        for i, ch in enumerate(children):
            result += assign_bbox(ch)
        return result
    for ch, (x, y, w, h) in zip(children, rules):
        if isinstance(ch, str):
            result.append((ch, x, y, w, h))
        else:
            result += assign_bbox(ch)
    return result

# 轉為 YOLO 格式
def to_yolo(class_id, x, y, w, h):
    return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

# 載入 char2id 映射
def get_char2id(char2id_path):
    char2id = {}
    with open(char2id_path, "r", encoding="utf-8") as f:
        for line in f:
            ch, id_str = line.strip().split('\t')
            char2id[ch] = int(id_str)
    return char2id

# 路徑設定
root_dir = Path("D:/chenm/nycu/AI/AI_Final/yolo")
myids_path = root_dir / "train_dataset/myids.txt"
output_path = root_dir / "train_dataset/labels"
char2id_path = root_dir / "train_dataset/char2id.txt"

char2id = get_char2id(char2id_path)

# 處理 myids.txt 並寫出 label
with open(myids_path, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        hanzi = parts[2]
        ids = parts[3]
        try:
            tree = parse_ids(ids)
            bboxes = assign_bbox(tree)

            with open(output_path / f"{hanzi}.txt", "w", encoding="utf-8") as out:
                for ch, x, y, w, h in bboxes:
                    if ch not in char2id:
                        print(f"⚠️ 字元 {ch} 沒在 char2id.txt 中，跳過")
                        continue
                    class_id = char2id[ch]
                    out.write(to_yolo(class_id, x, y, w, h) + "\n")
        except Exception as e:
            print(f"[跳過] {hanzi}: {ids} ({e})")
