# Handwritten Alphabet Classification

## Overview
This project involves classifying handwritten alphabets (A-Z) using machine learning and deep learning techniques. 
The primary goal is to analyze and preprocess the dataset, implement multiple models, and compare their performance. 
The final step involves determining the best-performing model to classify letters accurately.

---

## Dataset
The dataset used for this project is the **Handwritten Alphabets Dataset**. It consists of:
- Greyscale images of size 28x28 pixels.
- 26 classes representing the English alphabets (A-Z).


---

## Features
This project is divided into several stages, as outlined below:

### 1. **Data Exploration and Preparation**
- **Distribution Analysis:** Identify the number of unique classes and analyze their distribution in the dataset.
- **Normalization:** Scale pixel values for each image to a normalized range (e.g., [0, 1]).
- **Reshaping:** Flattened vectors are reshaped to reconstruct and display the corresponding images during testing.

### 2. **Experiments and Results**
The project comprises three major experiments, each showcasing different techniques:

#### **Experiment 1: Support Vector Machines (Scikit-learn)**
- Train and evaluate **two SVM models**:
  - Linear Kernel
  - Nonlinear Kernel
- Provide:
  - Confusion matrix
  - Average F1-scores on the testing dataset.

#### **Experiment 2: Logistic Regression (From Scratch)**
- **One-vs-All Multi-Class Classification:**
  - Implement logistic regression from scratch.
  - Train and evaluate the model.
  - Plot error and accuracy curves for both training and validation datasets.
  - Provide:
    - Confusion matrix
    - Average F1-scores on the testing dataset.

#### **Experiment 3: Neural Networks (TensorFlow)**
- Design and train **two neural networks** with varying:
  - Number of hidden layers
  - Neurons
  - Activation functions
- Evaluate and compare the models:
  - Plot error and accuracy curves for training and validation datasets.
- Save and reload the best-performing model.
- Test the saved model with:
  - The testing dataset.
  - Images representing the alphabetical letters of team members' names.
- Provide:
  - Confusion matrix
  - Average F1-scores for the testing dataset.

### 3. **Comparison and Recommendation**
- Compare results from all three experiments.
- Highlight the best-performing model based on performance metrics and generalization capability.

---

## Output
- Classification results for the testing dataset.
- Performance metrics, including:
  - Confusion matrices
  - Average F1-scores
  - Accuracy and error plots for training and validation datasets.
- A trained model capable of recognizing handwritten alphabets.
- Demonstrations of the model’s ability to classify letters forming the names of team members.

---
## Contributors

We would like to thank the following contributors to this project:

- [**Shahd Osama**](https://github.com/shahdosama10).
- [**Shahd Mostafa**](https://github.com/ShahdMostafa30).
- [**Maryam Osama**](https://github.com/maryamosama33).
- [**Ahmed Saad**](https://github.com/ahmedsaad123456).
- [**Seif Ibrahim**](https://github.com/Seif-Ibrahim1).

---

Feel free to contribute to this project by opening issues or submitting pull requests.
