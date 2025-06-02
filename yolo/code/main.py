from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    #model = YOLO("D:/chenm/nycu/AI/AI_Final/runs/detect/ids-yolo-v8/weights/last.pt")
    model.train(
        data="D:/chenm/nycu/AI/AI_Final/yolo/dataset/data.yaml",
        imgsz=50,
        epochs=10,
        batch=32,
        name="ids-yolo-v8",
        device="cpu"  # 可改成 "cpu" 若 GPU 有問題
        #resume=True  
    ) 
  

if __name__ == "__main__":
    main()

# from ultralytics import YOLO

# model = YOLO("D:/chenm/nycu/AI/AI_Final/runs/detect/ids-yolo-v8/weights/best.pt")  # 或 last.pt

# metrics = model.val(
#     data="D:/chenm/nycu/AI/AI_Final/yolo/dataset/data.yaml",
#     split="test",           # ← 指定用 test 資料驗證
#     imgsz=50,              # 測試圖片大小，與訓練時一致
#     batch=32,
#     device="cpu"         # 可改成 "cpu"
# )

# if __name__ == "__main__":
    
#     print(metrics)
#     # 將 metrics 中的主要數值轉成 dict
#     results = {
#         "precision": metrics.box.precision,
#         "recall": metrics.box.recall,
#         "mAP50": metrics.box.map50,
#         "mAP50-95": metrics.box.map
#     }

#     # 儲存為 JSON
#     with open("yolo_test_metrics.json", "w") as f:
#         json.dump(results, f, indent=4)



