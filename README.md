# Phase_4_Group1_NLP_Project

# Twitter Sentiment Analysis on Apple & Google

This project performs **sentiment analysis** on tweets related to **Apple** and **Google**, using **Natural Language Processing (NLP)** techniques. It leverages **NLTK** for tokenization and preprocessing, and uses machine learning models (such as **LinearSVC with TF-IDF**) to classify tweets into **positive, neutral, or negative** sentiments.

---

## Project Structure

├── Phase4_NLP_Twitter_Sentiment_Apple_Google.ipynb # Main Jupyter Notebook
├── judge-1377884607_tweet_product_company.csv # Dataset
├── README.md # Project Documentation


---

## Dataset

- **File:** `judge-1377884607_tweet_product_company.csv`  
- The dataset contains tweets about **Apple** and **Google**, labeled with sentiment categories:
  - `positive`
  - `neutral`
  - `negative`

---

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Twitter-Sentiment-Apple-Google.git
   cd Twitter-Sentiment-Apple-Google

pip install -r requirements.txt
import nltk
nltk.download('punkt')
nltk.download('stopwords')


jupyter notebook Phase4_NLP_Twitter_Sentiment_Apple_Google.ipynb


# Twitter Sentiment Analysis: Apple vs Google

This project performs sentiment analysis on tweets related to Apple and Google, classifying them into sentiment categories using NLTK preprocessing and LinearSVC with TF-IDF features.

## Main Steps
 ### 1️. Load Dataset

Load the dataset into a Pandas DataFrame.

Dataset: judge-1377884607_tweet_product_company.csv

### 2️. Preprocessing with NLTK

Tokenization

Stopword Removal (while keeping negations such as not, no, n't)

Lowercasing

(Optional) Lemmatization with WordNetLemmatizer

### 3️. Vectorization

Apply TF-IDF Vectorization using a custom NLTK tokenizer

Features include unigrams and bigrams

### 4️. Model Training

Train a LinearSVC classifier on the training split

Validate performance on the validation split

### 5️. Evaluation

Classification Report: Precision, Recall, F1-score, and Support

Confusion Matrix Visualization to highlight performance




# Technologies Used

Python 3.9+

Pandas, NumPy

NLTK

Scikit-learn

Matplotlib


# Results

Classification Report: Shows precision, recall, and F1-score for each sentiment class.

Confusion Matrix: Provides a visual overview of classification accuracy.

