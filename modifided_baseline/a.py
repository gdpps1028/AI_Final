import os
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), 'CNN.csv')

# 1. 确认文件存在
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} 不存在！")

# 2. 确认文件大小 > 0
size = os.path.getsize(CSV_PATH)
print(f"CSV 檔案大小：{size} bytes")
if size == 0:
    raise ValueError("CSV 是空檔案，应该有内容！")

# 3. 尝试用 pandas 读回
df = pd.read_csv(CSV_PATH)
print(f"CSV 抓到 {len(df)} 列資料，欄位：{list(df.columns)}")
# 可视化前几行
print(df.head())
