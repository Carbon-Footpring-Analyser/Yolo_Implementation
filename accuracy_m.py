import matplotlib.pyplot as plt
import cv2

# Paths to saved plots
plot_files = [
    "runs/train/exp_medium3/results.png",
    "runs/train/exp_medium3/F1_curve.png",
    "runs/train/exp_medium3/P_curve.png",
    "runs/train/exp_medium3/R_curve.png",
    "runs/train/exp_medium3/confusion_matrix.png"
]

# Display each plot
for file in plot_files:
    img = cv2.imread(file)
    if img is not None:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(file.split("/")[-1])
        plt.show()
    else:
        print(f"Could not load {file}")
