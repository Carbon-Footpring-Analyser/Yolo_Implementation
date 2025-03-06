import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Start timer
start_time = time.time()

# Train the model and store results
results = model.train(
    data='data.yaml', 
    epochs=50, 
    imgsz=640, 
    batch=16, 
    project='runs/train', 
    name='exp',
    plots=True  # Enables built-in plotting
)

# End timer
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Extract training metrics
metrics = results.metrics  # Training metrics

# Plot train/val accuracy
plt.figure(figsize=(12, 5))
plt.plot(metrics['train/box_loss'], label='Train Box Loss', linestyle='--')
plt.plot(metrics['val/box_loss'], label='Val Box Loss')
plt.plot(metrics['train/cls_loss'], label='Train Cls Loss', linestyle='--')
plt.plot(metrics['val/cls_loss'], label='Val Cls Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Evaluate model on test set
test_results = model.val()

# Extract test metrics
test_metrics = test_results.metrics

# Plot test accuracy and loss
plt.figure(figsize=(12, 5))
plt.bar(['mAP50', 'mAP50-95'], [test_metrics['metrics/mAP_50'], test_metrics['metrics/mAP_50-95']], color=['blue', 'green'])
plt.xlabel('Metric')
plt.ylabel('Value')
plt.title('Test Accuracy Metrics')
plt.show()