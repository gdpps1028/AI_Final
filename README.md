# AI_Final

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

# 測試版網頁執行
```
cd Related_stroke_code
```

Run the code
```
python server.py
http://127.0.0.1:5000
```