# Cervical Cancer Risk Prediction Using Machine Learning

This repository contains machine learning experiments for cervical cancer
risk prediction, including exploratory data analysis, preprocessing,
dimensionality reduction, class-imbalance handling, model development
and classifier fusion.

**Goal:** Develop a reliable predictive model that addresses class imbalance, reduces feature dimensionality and improves accuracy using a fused machine learning classifier.

## Research Context

This work was developed as part of my Master's research project:

**Enhancing Cervical Cancer Risk Prediction by Fused Machine Learning Classifier**

The project investigates the use of machine learning methods to support risk prediction from cervical cancer-related clinical and questionnaire
features.

## Objective
- Handle **class imbalance** to ensure a more reliable predictive model.  
- Apply **dimensionality reduction** techniques (e.g., PCA) to streamline features, enhance model efficiency and reduce computational time.  
- Utilize a **fused machine learning classifier** to improve overall predictive performance and accuracy in identifying risk factors.
---
**Concise summary:**  
> “This project predicts cervical cancer risk factors by handling class imbalance, reducing feature dimensionality with PCA, and employing a fused machine learning classifier for improved accuracy.”

---
## ⚙️ Requirements

To run this project, install the following:

---
pandas
numpy
matplotlib
seaborn
scikit-learn

## Dataset
| Feature                               | Description                                         |
| ------------------------------------- | --------------------------------------------------- |
| **Demographics**                      |                                                     |
| Age                                   | Age of the patient (years)                          |
| Number of sexual partners             | Total number of sexual partners                     |
| First sexual intercourse              | Age at first sexual intercourse                     |
| Number of pregnancies                 | Total number of pregnancies                         |
| **Behavioral/Habit**                  |                                                     |
| Smokes                                | Smoker (yes=1/no=0)                                 |
| Smokes (years)                        | Number of years smoking                             |
| Hormonal contraceptives               | Use of hormonal contraceptives (yes=1/no=0)         |
| Hormonal contraceptives (years)       | Duration of use in years                            |
| IUD                                   | Use of intrauterine device (yes=1/no=0)             |
| IUD (years)                           | Duration of IUD use in years                        |
| **Medical History**                   |                                                     |
| STD                                   | History of sexually transmitted disease (yes/no)    |
| STD (number)                          | Number of STD diagnoses                             |
| Dx: condylomatosis                    | Diagnosis of condylomatosis (yes/no)                |
| Dx: cervical condylomatosis           | Diagnosis of cervical condylomatosis (yes/no)       |
| Dx: vulvo-perineal condylomatosis     | Diagnosis of vulvo-perineal condylomatosis (yes/no) |
| Dx: syphilis                          | Diagnosis of syphilis (yes/no)                      |
| Dx: pelvic inflammatory disease (PID) | Diagnosis of PID (yes/no)                           |
| Dx: genital herpes                    | Diagnosis of genital herpes (yes/no)                |
| Dx: molluscum contagiosum             | Diagnosis of molluscum contagiosum (yes/no)         |
| **STD Related**                       |                                                     |
| STDs: Number of diagnosis             | Number of STD diagnoses                             |
| STDs: Time since first diagnosis      | Time (months) since first STD diagnosis             |
| STDs: Time since last diagnosis       | Time (months) since last STD diagnosis              |
| Dx: HIV                               | Diagnosis of HIV (yes/no)                           |
| Dx: HPV                               | Diagnosis of HPV (yes/no)                           |
| Dx: hepatitis B                       | Diagnosis of hepatitis B (yes/no)                   |
| Dx: hepatitis C                       | Diagnosis of hepatitis C (yes/no)                   |
| Dx: trichomoniasis                    | Diagnosis of trichomoniasis (yes/no)                |
| Dx: cervicitis                        | Diagnosis of cervicitis (yes/no)                    |
| **Target Variables**                  |                                                     |
| Hinselmann                            | Positive cervical cancer screening test (yes/no)    |
| Schiller                              | Positive Schiller test (yes/no)                     |
| Cytology                              | Positive cytology test (yes/no)                     |
| Biopsy                                | Positive biopsy test (yes/no)                       |

- **Source:** UCI Machine Learning Repository – “Cervical Cancer (Risk Factors)”. 
- **Size:** 858 instances (rows) × 36 features. 
- **Features:** A mix of demographic, behavioural/habit, and historic medical record variables. Examples include: Age, Number of sexual partners, Age of first sexual intercourse, Number of pregnancies, Smokes (yes/no), Years of smoking, Hormonal contraceptives (yes/no), Years using hormonal contraceptives, IUD (yes/no), Years with IUD, various STDs (yes/no), Time since first diagnosis, Time since last diagnosis, and target variables such as Hinselmann, Schiller, Cytology, Biopsy (all binary). 
UCI Machine Learning Repository
- **Target:** Classification of cervical cancer risk/diagnosis (binary indicators for multiple tests: Hinselmann, Schiller, Cytology, Biopsy).

**Data Preprocessing steps:**
- Handling missing values
- Encoding categorical features
- Feature scaling / normalization
- Train-test split

---

## Tools & Technologies
- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- Jupyter Notebook for exploratory analysis

---

## Methods / Models
Implemented and compared the following machine learning algorithms:
1. **Logistic Regression** – baseline model
2. **Random Forest** – tree-based ensemble
3. **Naïve bayes** – Probabilistic classification model
4. **Fused machine learning classifier** – Ensemble classification model

---

## Key Skills Demonstrated
- Data Cleaning and Exploration
- Feature Engineering
- Model Training and Evaluation
- Evaluation using appropriate metrics (accuracy, precision, recall, F1-score)
- Cross-validation for reliable performance assessment
- Visualization using Seaborn & Matplotlib
- Scikit-learn model workflow

---

## Results

## Model Performance Metrics

| Classifier | Class | Precision | Recall | F1-Score | Accuracy |
|------------|-------|----------|--------|----------|---------|
| Random Forest | 0 | 1.0000 | 0.9886 | 0.9943 | 0.9940 |
| Random Forest | 1 | 0.9877 | 1.0000 | 0.9938 | 0.9940 |
| Naïve Bayes | 0 | 0.8600 | 0.9100 | 0.8800 | 0.8600 |
| Naïve Bayes | 1 | 0.9000 | 0.8300 | 0.8600 | 0.8600 |
| Logistic Regression | 0 | 1.0000 | 0.9886 | 0.9943 | 0.9940 |
| Logistic Regression | 1 | 0.9877 | 1.0000 | 0.9938 | 0.9940 |
| Fused ML Classifier | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Fused ML Classifier | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

- **Best performing model:** [ Fused ML Model]
- **Metrics:**  
  - Accuracy: 100%  
  - Precision: 100%  
  - Recall: 100%  
  - F1-score: 100%  

**Visualization:**  
Correlation Heatmaps

Feature Distribution Plots

Model Accuracy,confusion matrix, ROC AUC  curve Comparison






