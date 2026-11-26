# McDonald's Customer Review Rating Prediction using Word2Vec + ANN

## Project Overview

This project focuses on predicting customer ratings from McDonald's reviews using Natural Language Processing (NLP) and Artificial Neural Networks (ANN).

The project uses **Word2Vec** to convert text reviews into numerical word embeddings and an **ANN model** to classify reviews into five rating categories.

Optuna is used for hyperparameter tuning to improve model performance.

## Project Objectives

- Analyze customer reviews and rating distributions.
- Perform text preprocessing and cleaning.
- Generate word embeddings using Word2Vec.
- Build an Artificial Neural Network for rating classification.
- Optimize ANN hyperparameters using Optuna.
- Evaluate model performance using classification metrics.
- Deploy the trained model using Streamlit.

## Dataset

- **Dataset:** McDonald's Customer Reviews
- **Total records after preprocessing:** 33,346
- **Input feature:** Cleaned customer reviews
- **Target variable:** Rating
- **Number of classes:** 5

### Rating Mapping

| Rating | Encoded Value |
|---|---:|
| 1 star | 0 |
| 2 stars | 1 |
| 3 stars | 2 |
| 4 stars | 3 |
| 5 stars | 4 |

### Train-Test Split

- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratified sampling: Yes

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Gensim
- Word2Vec
- TensorFlow / Keras
- Scikit-learn
- Optuna
- Streamlit

## Project Workflow

### 1. Data Collection and Exploration

- Load the McDonald's customer review dataset.
- Examine dataset dimensions and columns.
- Check missing values and duplicate records.
- Analyze rating distribution and review lengths.

### 2. Text Preprocessing

The following cleaning operations are performed:

- Convert text into lowercase.
- Remove URLs.
- Remove HTML tags.
- Remove special characters and numbers.
- Preserve apostrophes.
- Remove extra spaces.
- Remove empty reviews.

### 3. Word2Vec Word Embeddings

Word2Vec converts words into numerical vectors based on their surrounding context.

The model uses the following parameters:

| Parameter | Value |
|---|---:|
| Vector size | 100 |
| Window | 5 |
| Minimum count | 1 |
| Architecture | Skip-gram |
| Epochs | 10 |

The Skip-gram architecture is selected using `sg=1`.

Each review contains a variable number of words. Therefore, the word embeddings are averaged to create a fixed-size 100-dimensional review representation.

### 4. Artificial Neural Network

An ANN is developed using TensorFlow and Keras.

The network receives 100-dimensional Word2Vec review vectors and predicts one of five rating classes.

### 5. Hyperparameter Tuning using Optuna

Optuna is used to optimize the ANN model.

The following hyperparameters are explored:

- Number of hidden layers
- Number of neurons
- Dropout rate
- Learning rate
- Optimizer
- Batch size

Early Stopping is used during optimization to reduce unnecessary training.

### 6. Model Evaluation

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Test loss

## Model Performance

### Word2Vec + Optuna Tuned ANN

| Evaluation Metric | Score |
|---|---:|
| Test Accuracy | 64.83% |
| Test Loss | 0.8963 |
| Macro Precision | 0.64 |
| Macro Recall | 0.52 |
| Macro F1-score | 0.52 |
| Weighted F1-score | 0.61 |

### Classification Report

| Rating Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 1 star | 0.65 | 0.93 | 0.76 |
| 2 stars | 0.70 | 0.12 | 0.21 |
| 3 stars | 0.59 | 0.36 | 0.44 |
| 4 stars | 0.57 | 0.38 | 0.45 |
| 5 stars | 0.68 | 0.84 | 0.75 |

The model performs better on 1-star and 5-star reviews. It struggles to distinguish intermediate ratings, especially 2-star reviews.

## Streamlit Web Application

The trained Word2Vec and ANN models are integrated into a Streamlit application.

### Application Features

- Enter a customer review.
- Clean and tokenize the review.
- Generate Word2Vec embeddings.
- Predict the customer rating.
- Display prediction confidence.
- Display probability distribution across five rating classes.

## Repository Structure

```text
mcdonalds-review-rating-word2vec/
│
├── app.py
├── requirements.txt
├── final_ann_model.keras
├── word2vec_model.model
├── best_hyperparameters.json
└── README.md
```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/HruthwikThota/mcdonalds-review-rating-word2vec.git
```

### 2. Navigate to the Project Folder

```bash
cd mcdonalds-review-rating-word2vec
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

## Future Improvements

- Implement LSTM for sequence-based sentiment classification.
- Explore advanced NLP techniques.
- Compare TF-IDF, Word2Vec and LSTM models.
- Improve classification performance for intermediate ratings.
- Experiment with pretrained word embeddings.

## Acknowledgement

This project is part of my Natural Language Processing learning journey, focusing on text preprocessing, Word2Vec embeddings, Artificial Neural Networks and model evaluation.
