
from pathlib import Path

root_dir = Path("D:/chenm/nycu/AI/AI_Final")

radical_string = """一丨丶丿乙亅二亠人儿入八冂冖冫几凵刀力勹匕匚匸十卜卩厂厶又口囗土士夂夊夕大女子宀寸小尢尸屮山巛工己巾干幺广廴廾弋弓彐彡彳心戈戶手支攴文斗斤方无日曰月木欠止歹殳毋比毛氏气水火爪父爻爿片牙牛犬玄玉瓜瓦甘生用田疋疒癶白皮皿目矛矢石示禸禾穴立竹米糸缶网羊羽老而耒耳聿肉臣自至臼舌舛舟艮色艸虍虫血行衣襾見角言谷豆豕豸貝赤走足身車辛辰辵邑酉釆里金長門阜隶隹雨靑非面革韋韭音頁風飛食首香馬骨高髟鬥鬯鬲鬼魚鳥鹵鹿麥麻黃黍黑黹黽鼎鼓鼠鼻齊齒龍龜龠"""


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

def get_char2id():
    char2id = {}
    with open(root_dir / "yolo/train_dataset/char2id.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            ch, idstr = line.strip().split('\t')
            char2id[ch] = int(idstr)
    return char2id


yolo_script = "yolo detect train model=yolov8n.pt data=D:/chenm/nycu/AI/AI_Final/yolo/dataset/data.yaml imgsz=50 epochs=100 batch=32 name=ids-yolo-v8"

def labeled_character():
    #label_path = Path("AI_Final/yolo/train_dataset/label_txt/")
    label_path = root_dir / "yolo/train_dataset/label_txt"
    labeled_character_list = []

    for label_txt in label_path.iterdir():
        character = label_txt.stem
        labeled_character_list.append(character)
    
    return labeled_character_list

def generate_all_character():
    img_path = root_dir / "data_new"
    output_file_path = root_dir / "yolo/train_dataset/all_character.txt"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    character_list = []

    for dir in img_path.iterdir():
        for img in dir.iterdir():
            character_list.append(img.stem.split('_')[0])
    
    character_list = list(set(character_list))
    

    return character_list


    

# def generate_myids():
#     all_character_path = root_dir / "yolo/train_dataset/all_character.txt"

#     with open(all_character_path, "r", encoding="utf-8") as f:
#         characters = set(line.strip() for line in f if line.strip())

#     # Process the IDS file and extract lines where the character is in our set
#     input_path =  root_dir / "yolo/train_dataset/ids.txt"
#     output_path =  root_dir / "yolo/train_dataset/myids.txt"

#     count = 0
#     with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
#         for line in infile:
#             if line.startswith("#") or not line.strip():
#                 print("1")
#                 continue
#             parts = line.strip().split("\t")
#             if len(parts) >= 3:
#                 char = parts[1]
#                 if char in characters:
#                     count +=1
#                     num_description = len(parts) - 2
#                     check = 0
#                     for i in range(num_description):
#                         if 
                        

#                     outfile.write(line)
    
#     print(count)
import re
from pathlib import Path

def generate_myids():

    all_character_path = root_dir / "yolo/train_dataset/all_character.txt"
    input_path = root_dir / "yolo/train_dataset/ids.txt"
    output_path = root_dir / "yolo/train_dataset/myids.txt"

    circled_digits = [chr(cp) for cp in range(0x2460, 0x2474)]  # ①～⑳
    exclude_tokens = set(circled_digits + ['⿻'])

    # 讀入要保留的字
    with open(all_character_path, "r", encoding="utf-8") as f:
        characters = set(line.strip() for line in f if line.strip())

    count = 0
    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in infile:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            code, ch = parts[0], parts[1]
            if ch not in characters:
                continue

            descriptions = parts[2:]
            selected_descr = None

            for descr in descriptions:
                # 去掉 [xxx] 附加內容
                cleaned_descr = re.sub(r"\[.*?\]", "", descr)
                if not any(c in exclude_tokens for c in cleaned_descr):
                    selected_descr = cleaned_descr.strip()
                    break

            if selected_descr is None:
                selected_descr = ch

            outfile.write(f"{code}\t\t{ch}\t{selected_descr}\n")
            count += 1

    print(f"{count} characters written to myids.txt")


circled_digits = [chr(cp) for cp in range(0x2460, 0x2474)]
ids_char = [chr(cp) for cp in range(0x2ff0, 0x2fff)]
capital_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
brackets = ['[', ']']

def get_dependency_char(dependency,components,level, ch):
    if(dependency[ch] is not None):
        return
    #print(f"{any([(comp == '⿻') for comp in components[ch]])}    {any([(comp in circled_digits)] for comp in components[ch])}")
    #if any([(comp == '⿻') for comp in components[ch]]) or any([(comp in circled_digits)] for comp in components[ch]):
    if any(comp == '⿻' for comp in components[ch]) or any(comp in circled_digits for comp in components[ch]):

        #print("1")
        level[ch] = 0
        dependency[ch] = [ch]
        return
    if(len(components[ch])==1):
        print("1")
        level[ch] = 0
        dependency[ch] = [ch]
        return
    for comp in components[ch]:
        if comp not in circled_digits and comp not in ids_char and comp not in capital_letters and comp not in brackets:
            get_dependency_char(dependency, components,level, comp)

    dependency[ch] = components[ch]

    l = level[components[ch][0]]
    for comp in components[ch]:
        l = max(l, level[comp])
    level[ch] = l+1
    
    


def get_dependency():
    myids_path = root_dir / "yolo/train_dataset/myids.txt"
    ids_path = root_dir / "yolo/train_dataset/ids.txt"
    dependency_path = root_dir / "yolo/train_dataset/depend.txt"

    all_character_list = []
    my_character_list = []
    dependency = {}
    components = {}
    level = {}

    #get all character
    with open(ids_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            parts= line.strip().split('\t')
            if(len(parts) < 3):
                continue
            unicode, ch, descript = parts[0], parts[1], parts[2]
            all_character_list.append(ch)

    # get my character
    with open(myids_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            parts= line.strip().split('\t')
            if(len(parts) < 3):
                continue
            unicode, ch, descript = parts[0], parts[1], parts[2]
            my_character_list.append(ch)
    
    # initialize label, dependency, components
    with open(ids_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            parts= line.strip().split('\t')
            if(len(parts) < 3):
                continue
            unicode, ch, descript = parts[0], parts[1], parts[2]
            dependency[ch] = None
            #components[ch] =descript
            components[ch] = []
            #print(components[ch])
            level[ch] = None
            for comp in descript:
                if comp in circled_digits or comp in ids_char or comp in all_character_list:
                    components[ch].append(comp)
            #print(components[ch])
    for c in circled_digits: level[c] = 0
    for i in ids_char: level[i] = 0
    print(f"my char num : {len(my_character_list)}")
    for mych in my_character_list:
        get_dependency_char(dependency, components,level,mych)
    
    max_level = level[my_character_list[0]]
    for mych in my_character_list:
        max_level=max(level[mych], max_level)
    print(f"max level = {max_level}")

    count = 0
    for key, val in dependency.items():
        if val is not None:
            count+=1
    print(f"outer dependency count : {count-len(my_character_list)}")

    count = 0
    with open(ids_path, "r", encoding = "utf-8") as input, open(dependency_path, "w", encoding = "utf-8") as output:
        lines = input.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            if(len(parts) >=3):
                code, ch, description = parts[0], parts[1], parts[2]
                if dependency[ch] is not None:
                    count += 1
                    output.write(f"{parts[0]}\t\t")
                    output.write(f"{parts[1]}\t")
                    #output.write(f"{parts[2]}\t")
                    for d in description:
                        if d not in capital_letters and d not in brackets:
                            output.write(f"{d}")
                    output.write(f"\t")
                    output.write(f"{level[ch]}\n")

    print(count)



if __name__ == '__main__':
    #generate_myids()
    print(ord(u"丸"))