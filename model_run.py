from ultralytics import YOLO

model = YOLO('runs/train/exp4/weights/best.pt')
model.val(data='data.yaml')

model.predict(source='0', show=True, conf=0.75)  # Webcam; adjust source for video files