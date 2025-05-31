import os
import json

directory = '../data_new'
with open('char_to_stroke_Map.json', 'r', encoding='utf-8') as f:
    Map = json.load(f)

wrong_record = []

for name in os.listdir(directory):
    full_path = os.path.join(directory, name)
    basename = os.path.basename(full_path)
    word = basename.split('_')[0]

    if word in Map:
        stroke = Map[word]
    else:
        wrong_record.append(word)
        continue

    # 新資料夾名稱，加上筆劃數
    new_name = f"{name}_{stroke}"
    new_path = os.path.join(directory, new_name)

    if os.path.exists(new_path):
        print(f"[跳過] 已存在：{new_name}")
        continue

    # 重新更名資料夾
    os.rename(full_path, new_path)
    print(f"重新命名：{name} → {new_name}")

# 顯示錯誤記錄 正常來說是空的
print(f"\n共 {len(wrong_record)} 個字找不到筆劃：")
for s in wrong_record:
    print(s, end=' ')
