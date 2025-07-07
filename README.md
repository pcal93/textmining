
---
# 🚀 **Text Mining Framework: Political Document Classification** 🗳️

Welcome to **Text Mining** — a Python-based framework designed to automate the classification of political documents, particularly **Q\&A sessions** from parliamentary institutions. The aim of this project is to build a powerful, flexible, and fast framework that can be easily embedded into web applications. It’s built to save time, streamline the debugging process, and ultimately classify political content more efficiently.

**Authors**: Pietro Calabrese, Alberto Caruso

**Course**: Data Mining and Machine Learning - University Exam Project

---


## 📜 **Project Overview**

This project introduces a unique **Text Mining framework** to automate the classification of political documents, focusing on **Q\&A sessions** from different parliamentary institutions. This framework is designed to be user-friendly, efficient, and highly adaptable.

In this work, we used **supervised machine learning models** and tested 12 different algorithms based on two types of **text representation**:

* **Bag of Words (BOW)** 🧠 — Used with classical ML models
* **Word Embeddings (WE)** 🔍 — Used with **CNN** and **LSTM** models

We compare **classical learning algorithms** with **deep learning models** to provide a robust and scalable classification system for political text.

---

## 🗃️ **Dataset**

This project uses two main datasets to train and evaluate our models:

1. **Senato della Repubblica Italiana Dataset** 🇮🇹
   A collection of Q\&A sessions from the Italian Senate provided by the course instructor.

2. **Indian Parliamentary Q\&A Dataset** 🇮🇳
   A dataset containing Q\&A sessions from the Indian Parliament. [Link to dataset](https://www.kaggle.com/datasets/rajanand/rajyasabha)

These datasets offer a rich variety of parliamentary documents and provide the ideal challenge for our classification system.

---

## ⚙️ **Project Phases**

The development of this framework is broken into **three main phases**:

### 1. **Dataset Pre-processing**

* **Normalization**: Standardizing text formats for consistency.
* **Tokenization**: Breaking down text into smaller units (words).
* **Stopwords Filtering**: Removing irrelevant words like "the", "and", etc.
* **Stemming**: Reducing words to their root forms (e.g., "running" → "run").

### 2. **Text Representation**

* **Bag of Words (BOW)**: Converting the text into fixed-length vectors based on word frequency. Used with **classical machine learning models** (SVM, CNB, MLP, PAC).
* **Word Embeddings (WE)**: Utilizing pre-trained embeddings (like **Word2Vec** and **GloVe**) to capture deeper semantic meanings of words. Used with **CNN** and **LSTM** models.

### 3. **Modeling & Evaluation**

* We train 12 different models, including both **classical machine learning algorithms** and **deep learning models**:

  * **BOW + Classical ML Algorithms**: SVM, CNB, MLP, PAC
  * **WE + Deep Learning Models**: CNN, LSTM

---

## 🧠 **Models Used**

We experimented with **12 different machine learning models**. Here’s a quick rundown of the algorithms we tested:

* **Support Vector Machine (SVM)** 🧳
* **Complement Naive Bayes (CNB)** 📉
* **Multilayer Perceptron (MLP)** ⚙️
* **Passive Aggressive Classifier (PAC)** 💥
* **Convolutional Neural Network (CNN)** 🧠
* **Long Short-Term Memory (LSTM)** 📡

Each model was evaluated using both **Bag of Words (BOW)** and **Word Embeddings (WE)**, to identify the optimal solution for classifying political documents.

---

## 📊 **Evaluation Metrics**

To assess the performance of our models, we used key **evaluation metrics** to measure accuracy and classification quality:

* **Accuracy**: How often the model correctly classifies a document.
* **F1-Score**: The balance between precision and recall (a better measure when data is imbalanced).
* **Precision**: How many of the positive predictions were actually correct.
* **Recall**: How many of the actual positive cases were correctly predicted.

---

## 📦 **Libraries Used**

We leverage a variety of powerful Python libraries to implement our framework:

* **Pandas**: For efficient data manipulation and analysis.
* **NLTK**: For text preprocessing tasks (tokenization, stopwords filtering, stemming).
* **Scikit-learn**: For machine learning algorithms and model evaluation.
* **PyTorch**: For implementing deep learning models (CNN, LSTM).
* **Jupyter Notebook**: For an interactive development and debugging experience.
  
---

## 💻 **Usage**

Once the environment is set up, open the Jupyter Notebook and explore the following:

* **Pre-processing**: Get hands-on with the dataset and apply text normalization, tokenization, stopwords removal, and stemming.
* **Model Training**: Train the models using **BOW** and **Word Embeddings**, and compare their performance.
* **Evaluation**: Analyze the results using **accuracy**, **F1-score**, **precision**, and **recall** metrics to determine the best performing model.

The notebook is designed to be interactive and easy to follow, allowing you to see results as you go.


## 📝 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---
