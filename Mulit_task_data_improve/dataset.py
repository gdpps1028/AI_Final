# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset

class MutDataSet(Dataset):
    def __init__(self,
                 samples:          list,
                 index_to_char:    dict,
                 radical_to_index: dict,
                 transform=None,
                 train: bool = True):
        self.samples          = samples
        self.index_to_char    = index_to_char
        self.radical_to_index = radical_to_index
        self.transform        = transform
        self.train            = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, char_idx = self.samples[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)

        if self.train:
            # 從資料夾名稱取字、部首、筆畫
            folder = os.path.basename(os.path.dirname(img_path))
            ch, radical_s, stroke_s = folder.split('_')
            radical = self.radical_to_index[radical_s]
            stroke  = int(stroke_s)

            return img, char_idx, stroke, radical

        else:
            # 驗證／測試階段只需前兩項 (其餘填 0)
            return img, char_idx, 0, 0
