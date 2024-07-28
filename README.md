# 🎯 Graduation Thesis K20CLC - HCMUS
----------------
## Topic: Data Perturbation for Explainable AI in Semantic Role Labeling

## Lecture in charge
- **Name**: **Master. Tuan Nguyen Hoai Duc**
- **Description**: Lecturer at Ho Chi Minh University of Sciences. Ho Chi Minh. Vietnam.

# Member Introduction

| Id         | Name                  | 
|------------|-----------------------|
| 20127680   | Pham Thi Anh Phat     |

# Project Introduction

Proposal of the Smart Substitution method for data perturbation for word features in the explanation of biomedical semantic role labeling tasks.

# Project Objective: 
Propose a new method to evaluate the importance of features in explaining task models in a truthful and objective manner.

# Project Description:

## 1. Introduction

### 1.1. Background
- **Semantic Role Labeling (SRL)** is a task that aims to identify the predicate-argument structure of a sentence.
- **Explainable AI (XAI)** is a field of AI that aims to make AI models more understandable and interpretable.

### 1.2. Problem Statement
- The current perturbation methods, such as masking and deleting, are not truthful and can lead to underestimation or overestimation of the importance of features in the XAI in SRL task.

### 1.3. Proposed Solution
- **Smart Substitution**: A new method for data perturbation for word features in the explanation of biomedical SRL tasks.

## 2. Methodology

### 2.1. Smart Substitution
- **Input**: A sentence with a predicate and its arguments.
- **Output**: A sentence with a predicate and its arguments, where the words are replaced uniquely.
- **Process**: 
  - **Step 1**: Identify the predicate and its arguments.
  - **Step 2**: Replace the words with proposed substitutions.
  - **Step 3**: Evaluate the importance of the features in the SRL task.

## 3. Experiment

### 3.1. Dataset
- **PASBio+ 2022**: A dataset for the biomedical Predicate-Argument Structure.

### 3.2. Evaluation Metric
- **Brier Score**: A metric that measures the loss score of the SRL task.
- **Influence Score**: A metric that measures the influence of the features in the SRL task.
- **Relevance Score**: A metric that measures the relevance of the features in the SRL task.

### 3.3. Baseline Method
- **Masking, deleting**: The baseline method for data perturbation for word features in the explanation.

### 3.4. Results
- The experimental results of the project demonstrate that the proposed method provides suitable and understandable explanations, overcoming the limitations of current perturbation methods such as masking and deleting.