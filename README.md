# AI_Final - overview
Do the kinds of model to test the accaruacy of classify 4803 chinese word

## Requirements
Python 3.10.x
beautifulsoup4==4.13.4
bs4==0.0.2
certifi==2025.4.26
charset-normalizer==3.4.2
colorama==0.4.6
filelock==3.13.1
fsspec==2024.6.1
idna==3.10
Jinja2==3.1.4
MarkupSafe==2.1.5
mpmath==1.3.0
networkx==3.3
numpy==2.1.2
opencv-python==4.11.0.86
pandas==2.2.3
pillow==11.0.0
python-dateutil==2.9.0.post0
pytz==2025.2
requests==2.32.3
six==1.17.0
soupsieve==2.7
sympy==1.13.1
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchvision==0.20.1+cu121
tqdm==4.67.1
typing_extensions==4.12.2
tzdata==2025.2
urllib3==2.4.0

## How to run 
Change the directory
```
cd AI_Final
```
Run the file
```
python main.py
```
Arguments
```
'-m', '--Model' // C for Character, R for Radical, S for Stroke, RS for Radical+Stroke
'-t', '--Train' // Add this flag if you want to retrain the model
'-e', '--Eval'  // Test the model, input an interger for amount of testcases, default to 100
```

# Another stroke_based model
```
cd Related_stroke_code
```
Run the file
```
python train.py
```
Bonus (可以把epoch的每次loss 和 Acc自動做成圖表 於網站上呈現)
```
tensorboard --logdir=runs
```

# 網頁執行
```
cd website
```

Run the code
```
python server.py
http://127.0.0.1:5000
```
# Run Baseline CNN
```
cd AI_Final
python -m modifided_baseline.ch_main
```
