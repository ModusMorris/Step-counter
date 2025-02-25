# 🏃 Step Counter with CNN & Peak Detection
This project provides a step counter using both traditional peak detection and a Convolutional Neural Network (CNN) to improve accuracy. It processes accelerometer data from both hands, aligns it with video-based automated step counting, and learns to predict steps precisely.
## 🚀 Features
### ✅ Automated Ground Truth Extraction:
- MediaPipe-based step detection extracts step frames directly from video using pose tracking.
### ✅ Data Synchronization:
- Video and accelerometer data are aligned using clap detection (detected in audio waveform).
- The number of frames from the video is matched to accelerometer sampling frequency, ensuring precise step alignment.
### ✅ Peak Detection (Baseline):
- Traditional ENMO-based peak detection for quick step estimation.
### ✅ Deep Learning-Based Step Counter:
- A CNN trained on ENMO features detects steps more reliably than peak detection.
### ✅ Interactive Visualization:
- Acceleration data, step peaks, and CNN predictions are plotted dynamically.


## 📂 Data Processing Workflow
Before training the CNN, data is preprocessed automatically and checked manuelly:
### **Step 1: Extract Video-Based Steps (Ground Truth)**
- Each video is analyzed using MediaPipe to detect ankle and foot index motion.
- Steps are extracted, stored in step_counts.csv, then scaled to match the accelerometer sampling rate (scaled_step_counts.csv).

### **Step 2: Synchronize Video & Accelerometer Data**
- The start and end points of accelerometer data are synchronized with the video using clap detection (clap_detection_methods.py).
- Acceleration data is sliced to match the segment of the video using save_acc_metadata_sliced.py.
  
### **Step 3: Compute ENMO & Normalize**
- ENMO (Euclidean Norm Minus One) is computed from X, Y, Z acceleration.
- Values are normalized for stable training.
- ![image](https://github.com/user-attachments/assets/70922138-f21e-4c10-9b71-af7890679f9d)

### **Step 4: Peak Picker Baseline**
- A simple peak detection is applied `peak_picker.py` for quick validation before CNN training.

## 🔍 Peak Detection (Baseline Approach)
The first step in detecting walking patterns is using a simple peak-picking algorithm.
This method detects peaks in the ENMO signal, identifying potential step events.
- is more effizent
- only checks peaks
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

## 🛠 How to Use
### 1 Set Up Environment
To ensure all dependencies are installed, choose one of the two methods:
**Option 1: Install with `requirements.txt`**
- `pip install -r requirements.txt`
**Option 2: Install via `environment.yml` (for Conda users)
- `conda env create -f environment.yml`
- `conda activate stepcounter`

### 2 Preprocess Data (Extract & Sync Ground Truth)
`python save_acc_metadata_sliced.py`
-> Extracts step frames from video & syncs with accelerometer data.

### 3 Run Peak Detection Baseline
`python peak_picker.py`
-> Detects steps using a simple peak detection algorithm.

### 4 Train CNN Model
`python training.py`
-> Trains the CNN on synchronized ENMO features.

## 🎯 Next Steps
- ✅ Refine CNN training to better align detected steps exactly on peaks.
- 🔍 Optimize CNN loss function for better step probability calibration.
- 📊 Experiment with larger datasets (different walking speeds).
- 🛠 Improve peak detection pre-filtering before CNN classification.


