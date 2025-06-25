# ==============================================================================
#  Machine Learning Project: Girlfriend Acquisition Likelihood Predictor
# ==============================================================================
#  A classification model to tackle a B.Tech student's most pressing problem.
#  Author: Abhin Krishna
#  Project for GitHub: https://github.com/dearabhin/girlfriend-predictor
# ==============================================================================

# --- Step 1: Import Necessary Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# --- Step 2: Generate a Realistic Synthetic Dataset ---


def generate_synthetic_data(num_samples=1000):
    """
    Generates a synthetic dataset based on relatable college-life features.
    The core logic is a weighted formula to determine the 'girlfriend_status',
    which makes the classification problem solvable and intuitive.
    """
    # Feature Generation
    events_attended = np.random.randint(0, 25, size=num_samples)
    canteen_hours = np.random.uniform(0, 20, size=num_samples)
    github_to_instagram_ratio = np.random.uniform(0.1, 5.0, size=num_samples)
    reply_rate = np.random.uniform(0.05, 0.95, size=num_samples)
    uniform_wrinkle_score = np.random.uniform(0, 10, size=num_samples)
    dept_similarity_index = np.random.randint(0, 2, size=num_samples)
    friend_zone_flag = np.random.randint(0, 2, size=num_samples)
    crush_eye_contact_count = np.random.choice(
        [0, 1, 2, 3, 4, 5], size=num_samples, p=[0.85, 0.05, 0.04, 0.03, 0.02, 0.01])

    # Target Variable Logic (The "Secret Sauce")
    score = (
        (events_attended * 0.08) +
        (canteen_hours * 0.1) +
        (reply_rate * 2.5) +
        (crush_eye_contact_count * 0.5) +
        (dept_similarity_index * 0.5) -
        (uniform_wrinkle_score * 0.2) -
        (friend_zone_flag * 3.0) -
        (np.abs(github_to_instagram_ratio - 1.2) * 0.5)
    )

    # Convert score to probability using the sigmoid function
    probability = 1 / (1 + np.exp(-score))

    # Determine girlfriend_status based on the probability
    girlfriend_status = (probability > np.random.uniform(
        0, 1, size=num_samples)).astype(int)

    # Create a DataFrame
    df = pd.DataFrame({
        'events_attended': events_attended, 'canteen_hours': canteen_hours,
        'github_to_instagram_ratio': github_to_instagram_ratio, 'reply_rate': reply_rate,
        'uniform_wrinkle_score': uniform_wrinkle_score, 'dept_similarity_index': dept_similarity_index,
        'friend_zone_flag': friend_zone_flag, 'crush_eye_contact_count': crush_eye_contact_count,
        'girlfriend_status': girlfriend_status
    })
    return df

# --- Step 3: Exploratory Data Analysis (EDA) ---


def perform_eda(dataframe):
    """
    Performs and visualizes key aspects of the dataset.
    """
    print("\n--- Starting Exploratory Data Analysis (EDA) ---")
    sns.set_style("whitegrid")

    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    corr = dataframe.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.show()

    # Distribution of the target variable
    plt.figure(figsize=(6, 4))
    sns.countplot(x='girlfriend_status', data=dataframe)
    plt.title('Distribution of Girlfriend Status (0=No, 1=Yes)')
    plt.show()

# --- Step 4: Model Training and Prediction Pipeline ---


def train_and_evaluate(dataframe):
    """
    Handles data preprocessing, model training, and evaluation.
    """
    # Define features (X) and target (y)
    X = dataframe.drop('girlfriend_status', axis=1)
    y = dataframe['girlfriend_status']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and Train Model
    print("\n--- Training a Logistic Regression Model ---")
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    print("Model training complete!")

    # Evaluate Model
    print("\n--- Evaluating Model Performance ---")
    y_pred = model.predict(X_test_scaled)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                'No GF', 'Got GF'], yticklabels=['No GF', 'Got GF'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return model, scaler, X.columns

# --- Step 5: The Simulator ---


def predict_my_chances(stats, model, scaler, feature_columns):
    """
    Takes a dictionary of personal stats, scales them, and predicts the outcome.
    """
    input_df = pd.DataFrame([stats])
    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    result_text = "YES! The model predicts a high probability." if prediction[
        0] == 1 else "Not yet. The model predicts a low probability."

    print("\n" + "="*40)
    print("      YOUR PERSONALIZED PREDICTION")
    print("="*40)
    print(f"Based on your stats, the prediction is: {result_text}")
    print(
        f"Probability [No, Yes]: [{prediction_proba[0][0]:.2f}, {prediction_proba[0][1]:.2f}]")
    print("="*40)


# --- Main Execution Block ---
if __name__ == '__main__':
    # Generate the data
    df = generate_synthetic_data(num_samples=500)
    print("--- Dataset Head ---")
    print(df.head())

    # Perform EDA
    perform_eda(df)

    # Train the model and get necessary components for prediction
    trained_model, data_scaler, feature_names = train_and_evaluate(df)

    # --- YOUR TURN: Fill in your own stats here! ---
    your_current_stats = {
        'events_attended': 10,
        'canteen_hours': 8,
        'github_to_instagram_ratio': 2.5,
        'reply_rate': 0.3,
        'uniform_wrinkle_score': 6,
        'dept_similarity_index': 1,
        'friend_zone_flag': 0,
        'crush_eye_contact_count': 0
    }

    # Run the simulator
    predict_my_chances(your_current_stats, trained_model,
                       data_scaler, feature_names)

    print("\nDisclaimer: This is a statistical model. Real life is wonderfully unpredictable!")
