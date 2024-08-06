import wandb
from ultralytics import YOLO

# Initialize Weights & Biases
wandb.login()

# Path to our trained model weights
weights_path = '/Users/karthikeyanm/Downloads/yolov8n.pt'

# Creating a new YOLO Model from the trained weights
model = YOLO(weights_path)

# Evaluate on the validation dataset
val_results = model.val(data='/Users/karthikeyanm/Downloads/data.yaml')
print(val_results)
