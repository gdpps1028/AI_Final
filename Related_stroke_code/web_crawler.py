import os
import shutil
import requests
from bs4 import BeautifulSoup
import json
import time

source_dir = '../data_character'
target_dir = '../strokes_data'
cache_file = 'char_to_stroke_Map.json'
DELAY = 0.1

# ==== 載入快取 ====
if os.path.exists(cache_file):
    with open(cache_file, 'r', encoding='utf-8') as f:
        stroke_dict = json.load(f)
else:
    stroke_dict = {}

# ==== 查筆劃數（含快取） ====
def get_strokes_from_moedict(char):
    if char in stroke_dict:
        print(f"[快取] {char}")
        return stroke_dict[char]

    url = f"https://pedia.cloud.edu.tw/Entry/Detail/?title={char}&search={char}"
    try:
        res = requests.get(url, timeout=5)
        res.encoding = 'utf-8'
    except Exception as e:
        print(f"[網路錯誤] {char}：{e}")
        return None

    soup = BeautifulSoup(res.text, 'html.parser')
    for li in soup.find_all('li'):
        field = li.find('span', class_='field')
        value = li.find('span', class_='value')
        if field and value and '總筆畫' in field.text:
            strong = value.find('strong')
            if strong:
                digits = ''.join(filter(str.isdigit, strong.text))
                if digits:
                    stroke = int(digits)
                    stroke_dict[char] = stroke
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(stroke_dict, f, ensure_ascii=False, indent=2)
                    print(f"[新增快取] {char} → {stroke} 畫")
                    return stroke
    return None

# ==== 主處理邏輯 ====
processed, skipped, failed = 0, 0, 0

for count, filename in enumerate(os.listdir(source_dir), start=1):
    char = filename
    full_path = os.path.join(source_dir, char)

    if not os.path.isdir(full_path):
        continue

    # 查筆劃（包含快取）
    strokes = get_strokes_from_moedict(char)
    if strokes is None:
        print(f"[失敗] 無法取得筆劃數：{char}")
        continue

    correct_path = os.path.join(target_dir, str(strokes), char)

    # 若已分類且路徑正確 → 跳過
    if os.path.exists(correct_path):
        if char not in stroke_dict:
            stroke_dict[char] = strokes
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(stroke_dict, f, ensure_ascii=False, indent=2)
            print(f"[快取] {char} → {strokes} 畫")
        print(f"[完成] {char} → {strokes} 畫（已分類+快取）")
        skipped += 1
        continue

    # 若分類錯誤 → 修正：移除錯誤分類
    for i in range(1, 36):
        wrong_path = os.path.join(target_dir, str(i), char)
        if i != strokes and os.path.exists(wrong_path):
            print(f"[修正] {char} 從 {i} 畫移到 {strokes} 畫")
            shutil.rmtree(wrong_path)

    # 複製到正確位置
    os.makedirs(correct_path, exist_ok=True)
    try:
        shutil.copytree(full_path, correct_path, dirs_exist_ok=True)
        print(f"[分類] {char} → {strokes} 畫 → {correct_path}")
    except Exception as e:
        print(f"[複製錯誤] {char}：{e}")
        continue

    if count % 100 == 0:
        print("暫緩一秒")
        time.sleep(1)
    else:
        time.sleep(DELAY)

print("\n分類完成")
