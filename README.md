# Girlfriend Acquisition Predictor: A Machine Learning Project

*A fun, educational classification project by Abhin Krishna to solve a B.Tech student's most pressing problem using data science.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HCgPxf3gdnqQyZgHvqao9dZHn53QUv7D?usp=sharing)

## 🧐 Overview

As a B.Tech student at Model Engineering College, I decided to apply machine learning to a critical classification problem: **Will I get a girlfriend in college?**

This project is a humorous yet practical application of data science principles. It takes a set of relatable, college-life features, builds a predictive model, and allows you to simulate your own chances. While the premise is lighthearted, the underlying concepts—data generation, exploratory data analysis, feature scaling, logistic regression, and model evaluation—are fundamental to any real-world machine learning task.

## ✨ Features

The model is trained on a synthetic dataset generated from the following features:

* `events_attended`: Number of workshops/tech fests attended.
* `canteen_hours`: Hours spent per week in the college canteen.
* `github_to_instagram_ratio`: A measure of productivity vs. social media presence.
* `reply_rate`: The proportion of replies received to "hi"s sent.
* `uniform_wrinkle_score`: An entropy score for physical appearance (0=sharp, 10=messy).
* `dept_similarity_index`: 1 if you share the same department, 0 otherwise.
* `friend_zone_flag`: A categorical label we all hope is 0.
* `crush_eye_contact_count`: A sparse but powerful feature.

## 🔬 Methodology & Concepts

The project follows a standard machine learning pipeline.

#### 1. Logistic Regression
This project uses **Logistic Regression**, a fundamental algorithm for binary classification. Unlike Linear Regression which predicts a continuous value, Logistic Regression predicts a probability that an instance belongs to a certain class. It uses the **Sigmoid function** to map any real-valued number into a value between 0 and 1.

$$P(Y=1) = \frac{1}{1 + e^{-z}}$$

Where `z` is the linear combination of input features. This makes it perfect for "Yes/No" questions like ours.

#### 2. Feature Scaling (`StandardScaler`)
Our features have vastly different scales (e.g., `events_attended` from 0-25 vs. `reply_rate` from 0-1). Models like Logistic Regression can be sensitive to this. **Standard Scaling** transforms the data so that it has a mean of 0 and a standard deviation of 1, ensuring all features contribute equally to the model's decision-making process.

#### 3. Model Evaluation
How do we know if our model is any good? We use several metrics:
* **Accuracy**: The percentage of correct predictions.
* **Precision**: Of all the times the model predicted "Yes", how often was it right?
* **Recall**: Of all the actual "Yes" cases, how many did the model correctly identify?
* **Confusion Matrix**: A table that summarizes the performance, showing True Positives, True Negatives, False Positives, and False Negatives.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/dearabhin/girlfriend-predictor.git](https://github.com/dearabhin/girlfriend-predictor.git)
    cd girlfriend-predictor
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the script:**
    ```bash
    python girlfriend_predictor.py
    ```
    The script will train the model, show evaluation plots, and run a prediction on a sample set of stats.

## 🤖 The Simulator
Open `girlfriend_predictor.py` and navigate to the bottom. You can change the values in the `your_current_stats` dictionary to reflect your own situation and see what the model predicts for you!

```python
# --- YOUR TURN: Fill in your own stats here! ---
your_current_stats = {
    'events_attended': 5,
    'canteen_hours': 10,
    # ... and so on
}