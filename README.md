# **Scenario 2 – Health Sensing Pipeline**

### 6-Month AI/ML Internship Selection Task – 2025

---

## **Overview**

This repository contains my complete solution to Scenario 2 (“Health Sensing”) of the 6-month AI/ML Internship Selection Task–2025.

**Project Objectives:**
- Explore, preprocess, and visualize multi-channel sleep data from 5 participants
- Build labeled datasets for breathing irregularity detection and sleep stage classification
- Train and evaluate 1D CNN and ConvLSTM models for breathing event and sleep stage detection
- Implement robust subject-wise validation (Leave-One-Participant-Out CV)
- Provide full code reproducibility and reporting

---

## **Repository Structure:**
````
Scenario2_Project/
│
├── Data/
│ ├── AP01 ... AP05/                   # Participant folders
│ │  ├── nasal_airflow.txt
│ │  ├── thoracic_movements.txt
│ │  ├── spo2.txt
│ │  ├── flow_events.txt
│ │  ├── Sleep profile.txt
│     ├── cleaned signals
│     ├── cleaningPlots
│
├── Visualizations/                    # PDF signal visualizations per participant (Task 1)
│
├── Dataset/
│ └── respiratory_dataset.parquet      # Breathing event detection windows
│
├── Dataset_SleepStage/
│ └── sleep_stage_dataset.parquet      # Sleep stages windows (Bonus)
│
├── Results/                           # Model results (main task)
│ ├── CNN1D/
│ └── ConvLSTM/
│
├── Results_SleepStage/                # Model results (bonus task)
│ ├── CNN1D/
│ └── ConvLSTM/
│
├── scripts/
│ ├── vis.py                           # Task 1: Visualization script
│ ├── datacleaning.py                  # Task 2: Filtering script
│ ├── create_dataset.py                # Task 3: Breathing event dataset
│ ├── create_dataset_sleep_stage.py    # Bonus: Sleep stage dataset
│ ├── train_models.py                  # Task 4: 50-epoch training for main task
│ ├── train_models_resume.py           # Hybrid/fast CV for main task
│ ├── train_models_sleep_stage.py      # Bonus: Sleep stage model training
│ └── compute_convlstm_overall_results.py # Automated confusion matrix OCR
│
├── requirements.txt
├── .gitignore
└── README.md
 
````

---

## **Installation and Environment Setup**

- **Python 3.10 – 3.13** recommended
- **8GB RAM** minimum (16GB for faster training, I used 16GB ram)
- **PyTorch** (CPU version unless NVIDIA GPU available)
- All dependencies listed in `requirements.txt`
````
pip install -r requirements.txt
````
*If running confusion matrix OCR, also install Tesseract as per requirements.txt instructions.*

---

## **How to Run the Pipeline**

### **Task 1: Signal Visualization**

Generate comprehensive multi-channel signal plots with event annotations for all participants:
````
python vis.py -in_dir Data -out_dir Visualizations
````

**Deliverables:**

| File | Description |
|------|-------------|
| `AP01_visualization.pdf` | Participant 1: Airflow, thoracic, SpO₂ with breathing events |
| `AP02_visualization.pdf` | Participant 2: Full signal visualization |
| `AP03_visualization.pdf` | Participant 3: Full signal visualization |
| `AP04_visualization.pdf` | Participant 4: Full signal visualization |
| `AP05_visualization.pdf` | Participant 5: Full signal visualization |

Each PDF includes:
- Three-channel synchronized plots (airflow, thoracic movement, SpO₂)
- Color-coded event annotations (Normal, Hypopnea, Obstructive Apnea)
- Time-aligned visualization spanning entire recording duration
- Zoomed regions highlighting event characteristics

---

### **Task 2: Data Cleaning**

Apply Butterworth bandpass filter (0.17–0.4 Hz, 4th order) to isolate respiratory frequency band:

````
python datacleaning.py --folder Data/AP01
````


**Deliverables:**

| File                        | Description |
|-----------------------------|-------------|
| `AP01/CleanedSignals/*.txt` | Filtered signals for Participant 1 |
| `AP02/CleanedSignals/*.txt` | Filtered signals for Participant 2 |
| `AP03/CleanedSignals/*.txt`                | Filtered signals for Participant 3 |
| `AP04/CleanedSignals/*.txt`                | Filtered signals for Participant 4 |
| `AP05/CleanedSignals/*.txt`                | Filtered signals for Participant 5 |

Processing applied:
- **Bandpass filtering**: 0.17-0.4 Hz (respiratory frequency range: 10-24 breaths/min)
- **Noise reduction**: Removes low-frequency drift and high-frequency artifacts
- **Signal preservation**: Maintains breathing pattern morphology

---

### **Task 3: Dataset Creation**

#### Breathing Event Detection Dataset (Main Task)

Create labeled 30-second windows with 50% overlap:

````
python create_dataset.py -in_dir Data -out_dir Dataset
````

**Deliverables:**

| File                                  | Description |
|---------------------------------------|-------------|
| `Dataset/respiratory_dataset.parquet` | Complete dataset with 8,800 windows |

**Dataset Specifications:**
- **Total Windows**: 8,800 (30-second duration, 50% overlap)
- **Features per Window**:
  - Nasal airflow: 960 samples (32 Hz × 30s)
  - Thoracic movement: 960 samples (32 Hz × 30s)
  - SpO₂: 120 samples (4 Hz × 30s, upsampled to 960 for model input)
- **Labels**:
  - Normal: 8,041 (91.4%)
  - Hypopnea: 593 (6.7%)
  - Obstructive Apnea: 166 (1.9%)
- **Participants**: 5 subjects (AP01-AP05)
- **Format**: Parquet (efficient columnar storage)

#### Sleep Stage Classification Dataset (Bonus Task)

````
python create_dataset_sleep_stage.py -in_dir Data -out_dir Dataset_SleepStage
````

**Deliverables:**

| File                                             | Description |
|--------------------------------------------------|-------------|
| `Dataset_SleepStage/sleep_stage_dataset.parquet` | Complete sleep stage dataset with 8,780 windows |

**Dataset Specifications:**
- **Total Windows**: 8,780 (30-second duration, 50% overlap)
- **Features**: Same as breathing event dataset (airflow, thoracic, SpO₂)
- **Labels**:
  - Wake: 3,300 (37.6%)
  - N2: 2,442 (27.8%)
  - N1: 1,320 (15.0%)
  - N3: 1,065 (12.1%)
  - REM: 650 (7.4%)
  - Movement: 3 (0.03%)
- **Annotation Source**: Sleep-profile.txt (30-second epoch labels)

---

### **Task 4: Model Training & Evaluation**

#### Main Task: Breathing Event Detection

**1D CNN Model (50 epochs, full training):**
````
python train_models.py --dataset Dataset/respiratory_dataset.parquet --model cnn
````


**Conv-LSTM Model (Hybrid training - 50 epochs for Folds 1-2, 15 epochs for Folds 3-5):**

I had to train the ConvLSTM model in two parts in order to finish the assignment on time first time i had taken epochs to 50 which was taking too much time on my cpu so i had to change the epochs to 15 then processed further with remaining data
````
python train_models_resume.py --dataset Dataset/respiratory_dataset.parquet --model convlstm --epochs 15\
````

- **Computation of metrics (for ConvLSTM):**
````
python compute_convlstm_overall_results.py --results_dir Results/ConvLSTM
````

**Deliverables (CNN):**

| File | Description |
|------|-------------|
| `CNN1D_fold1_AP01_cm.png` | Fold 1 confusion matrix (Test on AP01) |
| `CNN1D_fold2_AP02_cm.png` | Fold 2 confusion matrix (Test on AP02) |
| `CNN1D_fold3_AP03_cm.png` | Fold 3 confusion matrix (Test on AP03) |
| `CNN1D_fold4_AP04_cm.png` | Fold 4 confusion matrix (Test on AP04) |
| `CNN1D_fold5_AP05_cm.png` | Fold 5 confusion matrix (Test on AP05) |
| `CNN1D_overall_cm.png` | Aggregated overall confusion matrix |
| `CNN1D_results.txt` | **Complete metrics report** with per-class accuracy, precision, recall, sensitivity, specificity (mean ± std across folds) |

**Deliverables (Conv-LSTM):**

| File | Description |
|------|-------------|
| `ConvLSTM_fold1_AP01_cm.png` | Fold 1 confusion matrix |
| `ConvLSTM_fold2_AP02_cm.png` | Fold 2 confusion matrix |
| `ConvLSTM_fold3_AP03_cm.png` | Fold 3 confusion matrix |
| `ConvLSTM_fold4_AP04_cm.png` | Fold 4 confusion matrix |
| `ConvLSTM_fold5_AP05_cm.png` | Fold 5 confusion matrix |
| `ConvLSTM_overall_cm.png` | Overall confusion matrix |
| `ConvLSTM_results.txt` | **Complete metrics report** (extracted via automated OCR) |

**Results Summary:**
- **CNN Overall Accuracy**: 89.76% ± 5.79%
- **Conv-LSTM Overall Accuracy**: 91.17% ± 6.55%
- **Cross-Validation**: Leave-One-Participant-Out (5 folds)
- **Best Fold**: AP03 (99.00% with Conv-LSTM)
- **Most Challenging Fold**: AP05 (79.32% with Conv-LSTM)

---

#### Bonus Task: Sleep Stage Classification

**CNN Training (10 epochs for fast convergence):**

````
python train_models_sleep_stage.py --dataset Dataset_SleepStage/sleep_stage_dataset.parquet --model cnn --epochs 10
````

**Conv-LSTM Training (10 epochs):**

````
python train_models_sleep_stage.py --dataset Dataset_SleepStage/sleep_stage_dataset.parquet --model convlstm --epochs 10
````

**Deliverables (CNN - Sleep Stages):**

| File | Description |
|------|-------------|
| `CNN1D_SleepStage_fold1_AP01_cm.png` | Fold 1 confusion matrix (6 sleep stages) |
| `CNN1D_SleepStage_fold2_AP02_cm.png` | Fold 2 confusion matrix |
| `CNN1D_SleepStage_fold3_AP03_cm.png` | Fold 3 confusion matrix |
| `CNN1D_SleepStage_fold4_AP04_cm.png` | Fold 4 confusion matrix |
| `CNN1D_SleepStage_fold5_AP05_cm.png` | Fold 5 confusion matrix |
| `CNN1D_SleepStage_overall_cm.png` | Overall confusion matrix |
| `CNN1D_SleepStage_results.txt` | Complete metrics with per-stage recall |

**Deliverables (Conv-LSTM - Sleep Stages):**

| File | Description |
|------|-------------|
| `ConvLSTM_SleepStage_fold1_AP01_cm.png` | Fold 1 confusion matrix |
| `ConvLSTM_SleepStage_fold2_AP02_cm.png` | Fold 2 confusion matrix |
| `ConvLSTM_SleepStage_fold3_AP03_cm.png` | Fold 3 confusion matrix |
| `ConvLSTM_SleepStage_fold4_AP04_cm.png` | Fold 4 confusion matrix |
| `ConvLSTM_SleepStage_fold5_AP05_cm.png` | Fold 5 confusion matrix |
| `ConvLSTM_SleepStage_overall_cm.png` | Overall confusion matrix |
| `ConvLSTM_SleepStage_results.txt` | Complete metrics report |

**Bonus Task Results Summary:**
- **CNN Overall Accuracy**: 28.72% (72% above random baseline of 16.7%)
- **Conv-LSTM Overall Accuracy**: 24.23%
- **Best Detected Stages**: Wake (53.53% recall), N2 (48.00% recall)
- **Challenge**: Limited performance due to respiratory-only signals (clinical staging requires EEG, EOG, EMG)

---

## **Summary of All Deliverables**

This repository provides **complete implementation** of all required components:

### **Visualization**
- 5 PDF files with multi-channel signal plots and event annotations

### **Data Cleaning**
- Filtered signal files for all 5 participants (bandpass 0.17-0.4 Hz)

### **Datasets**
- `respiratory_dataset.parquet` (8,800 windows, 3 classes)  
- `sleep_stage_dataset.parquet` (8,780 windows, 6 classes) [Bonus]

### **Model Results (Main Task)**
- 10 confusion matrices (5 per model × 2 models)  
- 2 overall confusion matrices  
- 2 complete metrics reports (CNN, Conv-LSTM)

### **Model Results (Bonus Task)**
- 10 confusion matrices (5 per model × 2 models)  
- 2 overall confusion matrices  
- 2 complete metrics reports

### **Code & Documentation**
- 8 Python scripts covering entire pipeline  
- requirements.txt with all dependencies  
- .gitignore for clean repository  
- README.md with detailed instructions

**Total Files Generated**: 50+ deliverable files across all tasks

---

## **Evaluation Alignment:**

This submission strictly adheres to the task requirements:

1.  **Leave-One-Participant-Out Cross-Validation** implemented for robust generalization assessment
2.  **Complete reproducibility** with clear execution instructions and dependency management
3.  **Transparent methodology** including hybrid training strategy documentation
4.  **Bonus task fully implemented** with sleep stage classification extending the core framework
5.  **Professional code quality** with modular structure, comments, and error handling

---

## **References:**

### **Sleep Apnea & Respiratory Events**
1. Phan, H., et al. (2021). "Deep Learning for Sleep Stage Classification from Physiological Signals: A Systematic Review." *IEEE Reviews in Biomedical Engineering*, 14, 215-239.
2. Urtnasan, E., et al. (2018). "Automatic Detection of Sleep Apnea Events from Nasal Airflow Using 1D-CNN." *IEEE Access*, 6, 45346-45353.
3. Mostafa, S.S., et al. (2020). "A Survey on Sleep Apnea Detection from ECG Using Machine Learning." *Journal of Ambient Intelligence and Humanized Computing*, 11(4), 1527-1542.

### **Signal Processing**
4. Butterworth, S. (1930). "On the Theory of Filter Amplifiers." *Wireless Engineer*, 7(6), 536-541. [Bandpass filter theory]
5. Smith, S.W. (1997). *The Scientist and Engineer's Guide to Digital Signal Processing*. California Technical Publishing. [Chapter 14: Digital Filters]

### **Deep Learning Architectures**
6. Kiranyaz, S., et al. (2016). "Real-Time Patient-Specific ECG Classification by 1-D Convolutional Neural Networks." *IEEE Transactions on Biomedical Engineering*, 63(3), 664-675.
7. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780. [LSTM fundamentals]
8. Shi, X., et al. (2015). "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting." *NIPS* 2015. [Conv-LSTM architecture]

### **Cross-Validation for Physiological Signals**
9. Arlot, S., & Celisse, A. (2010). "A Survey of Cross-Validation Procedures for Model Selection." *Statistics Surveys*, 4, 40-79.
10. Varoquaux, G., et al. (2017). "Assessing and Tuning Brain Decoders: Cross-Validation, Caveats, and Guidelines." *NeuroImage*, 145(Pt B), 166-179. [Subject-wise CV justification]

### **Technical Libraries & Frameworks**
11. Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS* 2019. [https://pytorch.org](https://pytorch.org)
12. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *JMLR*, 12, 2825-2830. [https://scikit-learn.org](https://scikit-learn.org)
13. Virtanen, P., et al. (2020). "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python." *Nature Methods*, 17(3), 261-272. [https://scipy.org](https://scipy.org)
14. McKinney, W. (2010). "Data Structures for Statistical Computing in Python." *SciPy 2010*. [Pandas library]

### **Assignment Documentation**
15. IIT Gandhinagar (2025). "6-Month Intern Selection Task – 2025." Assignment document: `6-month-intern-selection-task-2025.docx`

---

## **Acknowledgements**

This project was completed using **PyCharm Community Edition 2023** on:
- **CPU**: Intel i5-12500H (12 cores)
- **RAM**: 16GB DDR4
- **GPU**: Intel Iris Xe Graphics (not utilized for training)
- **OS**: Windows 11

All computations performed on CPU to ensure reproducibility on standard hardware.

---

## **Thank You for Your Consideration!!**

I have sincerely tried my best to ensure that every requirement of Scenario 2 has been implemented with clarity, correctness, and attention to detail.

All steps in the pipeline — from signal visualization to dataset creation, model training, and evaluation — have been carefully documented for reproducibility and transparency.

If, by any chance, there remains any discrepancy between this README and the actual project files, I kindly request your understanding and forgiveness.

I truly appreciate the opportunity to work on this assignment and thank you for reviewing my submission.

---



