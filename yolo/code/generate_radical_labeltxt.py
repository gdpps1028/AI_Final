import os
from pathlib import Path
from supplement import radical_string, radical2id

# AI_Final
root_dir = Path("../../") 

img_directories = root_dir / "data_new"
label_dir = root_dir / "yolo/label"
label_dir.mkdir(parents=True, exist_ok=True)

radical_list = radical_string
radical2id = {r : i for i, r  in enumerate(radical_list)}
# print(radical2id)

radical_count = {r:0 for r in radical_list}

for img_dir in img_directories.iterdir():
    for img in img_dir.iterdir():

        if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        img_name = img.stem
        radical_name = img_name.split('_')[0]

        

        if radical_name not in radical2id:
            continue

        if radical_count[radical_name] == 1:
            continue
        radical_count[radical_name] = 1

        class_id = radical2id[radical_name]

        yolo_line = f"{class_id} 0.5 0.5 1.0 1.0\n"

        label_file = label_dir / f"{radical_name}.txt"
        with open(label_file, "w", encoding = "utf-8") as f:
            f.write(yolo_line)

for key, value in radical_count.items():
    if(value == 0):
        print(key)



