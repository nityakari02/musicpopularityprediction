# Music Popularity Prediction Project

## Overview

This project aims to predict the popularity of songs using machine learning techniques. By analyzing features such as loudness and danceability, we explore whether these attributes can accurately classify songs as popular or not. The project includes a comprehensive analysis using K-Nearest Neighbors (KNN) and Logistic Regression models to evaluate the predictive power of these features.

## Project Components

1. **Notebook: `Music Popularity Prediction Project.ipynb`**
   - A Jupyter Notebook containing the full analysis, data preparation, exploratory data analysis, model training, and evaluation.
   - The notebook includes detailed explanations and visualizations of the data and model performance.

2. **Data File: `song_data_orig.csv`** 
   - CSV file of the original data on song popularity from Spotify.
  
3. **PDF Report: `Music Popularity Prediction Project.pdf`**

   - It summarizes the key findings, methodology, and results of the project.

4. **Python Script: `Music Popularity Prediction Project.py`**
   - A Python script version of the notebook for quick execution and reproducibility.
   - It contains the essential code for data processing, model training, and evaluation.

## Key Features

- **Data Preparation:**
  - Cleaning and transforming the dataset to include relevant features.
  - Creating a binary classification label (`is_pop`) based on song popularity.

- **Exploratory Data Analysis (EDA):**
  - Visualizing the distribution of features and their relationships.
  - Identifying key attributes that correlate with song popularity.

- **Machine Learning Models:**
  - Implementing and comparing KNN and Logistic Regression models.
  - Evaluating models using accuracy, ROC curves, and AUC values.

- **Model Evaluation:**
  - Detailed comparison of model performance through confusion matrices and accuracy scores.
  - Discussion on the effectiveness of loudness and danceability in predicting song popularity.

## Skills and Technologies

### Data Analysis
- **Python Programming**
  - Libraries: `numpy`, `pandas`

### Data Visualization
- **Visualization Tools**
  - Libraries: `matplotlib`, `seaborn`

### Machine Learning
- **Model Implementation**
  - Algorithms: K-Nearest Neighbors (KNN), Logistic Regression
  - Libraries: `scikit-learn`
  - Functions: `train_test_split`, `StandardScaler`, `KNeighborsClassifier`, `LogisticRegression`, `roc_curve`, `auc`, `confusion_matrix`, `ConfusionMatrixDisplay`
- **Model Evaluation**
  - Metrics: Accuracy scores, ROC curves, AUC values, Confusion Matrices

### Data Preparation
- **Data Cleaning and Transformation**
  - Handling missing values
  - Feature selection and engineering
  - Data normalization and standardization

### Project Management
- **Development Tools**
  - Jupyter Notebook: For interactive data analysis and model development
  - Python Script: For reproducible code execution
- **Documentation**
  - PDF Report: Summarizes project findings and methodologies

### Tools
- **Jupyter Notebook**: Interactive computing environment for developing and presenting the project.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing.
- **Matplotlib**: Data visualization.
- **Seaborn**: Statistical data visualization.
- **Scikit-Learn**: Machine learning library for model implementation and evaluation.

## Conclusion

The Music Popularity Prediction Project demonstrates the application of machine learning to predict song popularity. Both KNN and Logistic Regression models show promising results, with specific features like loudness and danceability proving to be significant predictors. This project highlights the potential for using data-driven approaches to understand and predict trends in the music industry.
