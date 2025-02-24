# 🏃 Step Counter with CNN & Peak Detection
This project provides a step counter using both traditional peak detection and a Convolutional Neural Network (CNN) to improve accuracy. It processes accelerometer data and learns to predict steps more precisely.
## 🚀 Features
- 📊 Data Processing: Reads accelerometer data from left and right foot sensors, computes ENMO (Euclidean Norm Minus One) as a feature.
- ✅ Ground Truth Steps: Steps labeled from video data serve as the ground truth for model evaluation.
- 📉 Simple Peak Detection: Uses a basic peak-picking algorithm for quick step estimation.
- 🤖 CNN-Based Step Detection: A deep learning approach to learn step patterns and improve step counting.
- 📈 Visualization: Plots acceleration data, peak detections, and CNN predictions with interactive graphs (using plotly).

## 📂 Data Processing
The pipeline processes raw accelerometer data (X, Y, Z) and extracts step labels from csv automatically 
1. Compute ENMO:
   ![image](https://github.com/user-attachments/assets/70922138-f21e-4c10-9b71-af7890679f9d)
(Negative values are clipped to zero.)
2. Normalize Data: Standardizes ENMO values to ensure stable training.
3. Extract Ground Truth Steps:
  - Ground truth step frames are manually labeled from videos.
  - These are stored in scaled_step_counts.csv.

## 🔍 Peak Detection (Baseline Approach)
A simple peak-picking algorithm identifies step peaks using the accelerometer signal:
![image](https://github.com/user-attachments/assets/ffcb6006-58ee-4add-a479-c24470108a92)

## 🤖 CNN-Based Step Detection
A 1D CNN is trained on ENMO data to detect steps more accurately.

### Model Architecture
- Conv1D Layers extract temporal step features.
- Pooling Layers reduce noise.
- Fully Connected Layers output a probability for each frame.
- ✅ The CNN learns step patterns and improves detections by reducing false positives and misalignments.

## 📊 Evaluation & Visualization
- The model is tested on unseen data.
- Steps detected by the CNN (🔴 red dots) are compared to ground truth steps (crosses).
![CNN_ENMO](https://github.com/user-attachments/assets/a48e9b87-c2aa-4c5c-902b-81da4c016bbe)
