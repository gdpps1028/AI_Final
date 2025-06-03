import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskCNN(nn.Module):
    def __init__(self, num_classes: int, num_radicals: int):
        super(MultiTaskCNN, self).__init__()
        # same as baseline
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.pool2 = nn.MaxPool2d(2)
        feat_dim = 32 * 12 * 12

        # 主分支：中文字分類
        self.fc_char = nn.Linear(feat_dim, num_classes)

        # 輔助分支 A
        self.fc_stroke = nn.Linear(feat_dim, 1)

        # 輔助分支 B
        self.fc_radical = nn.Linear(feat_dim, num_radicals)

    def forward(self, x):
        x = self.pool1(self.dropout1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.pool2(self.dropout2(self.relu2(self.bn2(self.conv2(x)))))
        # Flatten
        x = x.view(x.size(0), -1)
        out_char    = self.fc_char(x)       # 分類 logits
        out_stroke  = self.fc_stroke(x)     # 回歸值
        out_radical = self.fc_radical(x)    # 部首 logits

        return out_char, out_stroke.squeeze(1), out_radical
    
    ## used for 分析
    def _forward_features(self, x):
        x = self.pool1(self.dropout1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.pool2(self.dropout2(self.relu2(self.bn2(self.conv2(x)))))
        x = x.view(x.size(0), -1)
        return x

    def extract_features(self, x):
        """提取 CNN encoder 的 flatten 特徵向量"""
        return self._forward_features(x)