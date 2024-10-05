# Document Object Detection
This project implements a RetinaNet model for detecting stamps and signatures in document images using PyTorch.
Features

Custom dataset for document images with stamps and signatures
RetinaNet model with ResNet50 backbone
Training and validation pipeline
Inference script for making predictions on new images
MLflow integration for experiment tracking

Setup

Install required packages:
Copypip install torch torchvision tqdm albumentations mlflow fvcore

Prepare your dataset:

Organize images in a directory
Create a JSONL file with annotations


Update configuration in the script:

Set BATCH_SIZE, RESIZE_TO, NUM_EPOCHS, etc.
Configure MLflow tracking URI and credentials



Usage
Training
Run the main script to start training:
Copypython main.py
This will:

Load and preprocess the dataset
Train the model
Log metrics to MLflow
Save checkpoints and best model

Inference
Use the inference script to make predictions on new images:
Copypython inference.py --image_path path/to/your/image.jpg
Model Architecture

Base: RetinaNet
Backbone: ResNet50 with FPN
Classes: Background, Stamp, Signature
