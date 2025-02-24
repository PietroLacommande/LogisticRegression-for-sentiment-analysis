# Mini Project 1: Movie Review Sentiment Analysis

This repository contains a **Logistic Regression**‐based sentiment analysis system for **movie reviews**, modeled after the steps in Chapters 1–6 and similar to the approach in Chapter 8.
---

## Table of Contents
1. [Project Overview and Objectives](#project-overview-and-objectives)  
2. [Dataset Acquisition and Preparation](#dataset-acquisition-and-preparation)  
3. [Text Preprocessing](#text-preprocessing)  
4. [Feature Extraction](#feature-extraction)  
5. [Model Training](#model-training)  
6. [Model Evaluation](#model-evaluation)  
7. [Hyperparameter Tuning](#hyperparameter-tuning)  
8. [Learning Curve Analysis](#learning-curve-analysis)  
9. [Carbon Footprint Analysis with CodeCarbon](#carbon-footprint-analysis-with-codecarbon)  
10. [Ethical Considerations and Model Explainability](#ethical-considerations-and-model-explainability)  
11. [Deployment on Embedded Systems](#deployment-on-embedded-systems)  
12. [Code Release Responsibilities](#code-release-responsibilities)  
13. [License](#license)  

---

## Project Overview and Objectives

The primary goal of this project is to build a **sentiment analysis system** using Logistic Regression on the **IMDb movie reviews dataset**. The key objectives are:

- **Preprocess** and **vectorize** text reviews.  
- **Train** and **evaluate** a logistic regression model for sentiment classification.  
- **Tune hyperparameters** to optimize performance.  
- **Measure** the **carbon footprint** of the model using CodeCarbon.  
- Address **ethical considerations**, including **explainability** and **fairness** in ML.  
- **Deploy** the trained model on an **embedded system** (Arduino) with limited memory and processing power.

---

## Dataset Acquisition and Preparation

1. **Download the IMDb movie review dataset**:  
   - Obtain it from [Stanford’s site](https://ai.stanford.edu/~amaas/data/sentiment/).  
   - Unzip the dataset, which typically provides a folder structure containing `train` and `test` sets, each with `pos` and `neg` subfolders.

2. **Load the data**:  
   - Read all text files from the `pos` folder as positive examples and from the `neg` folder as negative examples.  
   - Combine them into a single dataframe (or dataset) with two columns: **review** (text) and **sentiment** (label: 0 or 1).  

3. **Explore** the dataset:  
   - Check the distribution of classes (positive vs. negative).  
   - Identify potential data quality issues (duplicates, encoding, etc.).

---

## Text Preprocessing

To clean the text reviews:

1. **Remove HTML tags**, punctuation, or other artifacts.  
2. **Convert text to lowercase** for consistency.  
3. **Tokenize** the text into individual words.  
4. **Remove stop words** (common words that don’t contribute meaningful information).  

This process ensures that our model focuses on relevant terms and handles variations in word forms.

---

## Feature Extraction

Once the text is cleaned, transform it into numerical features that the machine learning model can understand:

- **TF‐IDF (Term Frequency‐Inverse Document Frequency)**:  
  - TF‐IDF captures how important a word is relative to its frequency in a given document and across the entire corpus.  
- Alternatively, other techniques like **Bag‐of‐Words** or advanced embeddings (e.g., **Word2Vec**, **GloVe**, **BERT**) could be explored. For this project, TF‐IDF is sufficient.

---

## Model Training

1. **Split** the data into **training** and **test** sets (50%/ 50%).  
2. **Instantiate** the `LogisticRegression` model (e.g., `LogisticRegression`).  
3. **Train** the model on the training data:  
   ```python
   log_reg.fit(X_train, y_train)

## Ethical Considerations and Model Explainability
Building ethical and transparent AI systems requires:

Explainable AI Techniques:

Use SHAP (SHapley Additive exPlanations) to see which words most heavily influence the model’s predictions.
This helps detect potential biases or unexpected behavior.
Fairness:

Check if certain types of reviews are misclassified more often.
Consider data diversity and whether reviews from different genres or languages might affect model performance.

## Deployment on Embedded Systems
For resource‐constrained devices such as Arduino:

Extract the final Logistic Regression model parameters (coefficients and bias).
Quantize them to fixed‐point integers (to remove floating‐point dependency).
Generate a .h header file with these parameters.
Integrate into an Arduino sketch:
Perform the inference using integer arithmetic.
Load the quantized weights and bias, run the computations on the microcontroller.
This TinyML approach allows you to run the model on devices with limited compute and memory.

## Licence

This project is licensed under the MIT License

## Responsibility and Code Release Practices

When releasing source code on GitHub, it is essential to follow responsible practices.
- Pull requests must be validated by at least one team member
- Unit tests should be ran automatically in a CI/CD pipeline
- Syntaxic tests with tools like SonarCloud should also be ran to validate the code
