# ==============================================================================
#  Machine Learning Project: Girlfriend Acquisition Likelihood Predictor (V3)
# ==============================================================================
#  A sophisticated model incorporating advanced social, environmental,
#  and real-world constraint features for a more holistic prediction.
#  Author: Abhin Krishna
#  Project for GitHub: https://github.com/dearabhin/girlfriend-predictor
# ==============================================================================

# --- Step 1: Import Libraries (No Changes) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# --- Step 2: Generate an Advanced Synthetic Dataset (V3) ---


# Increased sample size for more features
def generate_synthetic_data_v3(num_samples=2000):
    """
    Generates an advanced synthetic dataset with nuanced social and environmental features.
    """
    # --- Kept Features ---
    events_attended = np.random.randint(0, 30, size=num_samples)
    reply_rate = np.random.uniform(0.05, 0.95, size=num_samples)
    dept_similarity_index = np.random.randint(0, 2, size=num_samples)
    friend_zone_flag = np.random.randint(0, 2, size=num_samples)
    crush_eye_contact_count = np.random.choice(
        [0, 1, 2, 3, 4, 5], size=num_samples, p=[0.75, 0.1, 0.08, 0.04, 0.02, 0.01])
    communication_skill_score = np.random.uniform(2, 10, size=num_samples)

    # --- New & Advanced Features (V3) ---
    personal_grooming_score = np.random.uniform(
        3, 10, size=num_samples)  # Replaces wrinkle_score
    semester_cgpa = np.random.uniform(7.5, 9.8, size=num_samples)
    hackathon_participation_count = np.random.choice(
        [0, 1, 2, 3, 4], size=num_samples, p=[0.5, 0.3, 0.1, 0.05, 0.05])
    other_competition_wins = np.random.choice(
        [0, 1, 2], size=num_samples, p=[0.8, 0.15, 0.05])
    avg_dm_conversations_per_week = np.random.randint(0, 15, size=num_samples)
    provides_academic_help = np.random.randint(0, 2, size=num_samples)
    target_is_single = np.random.choice([0, 1], size=num_samples, p=[
                                        0.3, 0.7])  # 30% chance target is not single
    local_competition_score = np.random.uniform(
        1, 10, size=num_samples)  # 1-10 scale

    # --- Target Variable Logic (The "Secret Sauce" V3) ---
    # This formula is now much more complex.
    score = (
        (events_attended * 0.06) +
        (reply_rate * 1.5) +
        (crush_eye_contact_count * 0.3) +
        (communication_skill_score * 0.25) +
        (personal_grooming_score * 0.2) +      # Grooming matters
        (provides_academic_help * 0.5) +       # Being helpful is attractive
        (hackathon_participation_count * 0.2) +  # Participation is good
        (other_competition_wins * 0.4) +       # Winning is better
        (avg_dm_conversations_per_week * 0.1) +  # Direct contact helps
        # Assume ~8.8 CGPA is a social sweet spot
        (np.abs(semester_cgpa - 8.8) * -0.3) -
        (friend_zone_flag * 4.0) -             # Heavier penalty
        (local_competition_score * 0.2)      # Competition makes it harder
    )

    # The "Game-Changer" Constraint
    # If the target is not single, the chance of success plummets.
    # We apply a massive penalty to the score AFTER the main calculation.
    # Apply a huge penalty if target is not single
    score[target_is_single == 0] -= 10

    probability = 1 / (1 + np.exp(-score))
    girlfriend_status = (probability > np.random.uniform(
        0.4, 1, size=num_samples)).astype(int)

    # Ensure the constraint holds: if target is not single, status must be 0
    girlfriend_status[target_is_single == 0] = 0

    # Create DataFrame
    df = pd.DataFrame({
        'events_attended': events_attended, 'reply_rate': reply_rate,
        'dept_similarity_index': dept_similarity_index, 'friend_zone_flag': friend_zone_flag,
        'crush_eye_contact_count': crush_eye_contact_count, 'communication_skill_score': communication_skill_score,
        'personal_grooming_score': personal_grooming_score, 'semester_cgpa': semester_cgpa,
        'hackathon_participation_count': hackathon_participation_count, 'other_competition_wins': other_competition_wins,
        'avg_dm_conversations_per_week': avg_dm_conversations_per_week, 'provides_academic_help': provides_academic_help,
        'target_is_single': target_is_single, 'local_competition_score': local_competition_score,
        'girlfriend_status': girlfriend_status
    })
    return df


# --- Run the V3 Pipeline ---
df_v3 = generate_synthetic_data_v3()
print("--- V3 Dataset Head ---")
print(df_v3.head())

# The modeling pipeline function from V2 can be reused


def run_model_pipeline(dataframe):
    X = dataframe.drop('girlfriend_status', axis=1)
    y = dataframe['girlfriend_status']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print("\n--- Model Evaluation (V3) ---")
    y_pred = model.predict(X_test_scaled)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    return model, scaler, X.columns, X_test, y_test  # Return test sets for EDA


model_v3, scaler_v3, feature_columns_v3, X_test_v3, y_test_v3 = run_model_pipeline(
    df_v3)


# --- Step 4: The "Abhin Simulator" (V3) ---
def predict_my_chances_v3(my_stats, model, scaler, feature_columns):
    # This function doesn't need to change
    input_df = pd.DataFrame([my_stats])
    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)
    result_text = "YES! The model predicts a favorable outcome." if prediction[
        0] == 1 else "Not yet. The model sees significant obstacles."
    print(
        f"\n--- Prediction (V3) ---\nResult: {result_text}\nProbability [No, Yes]: [{proba[0][0]:.2f}, {proba[0][1]:.2f}]")


# --- YOUR TURN: Fill in your stats for the V3 model! ---
your_current_stats_v3 = {
    'events_attended': 15, 'reply_rate': 0.4,
    'dept_similarity_index': 1, 'friend_zone_flag': 0,
    'crush_eye_contact_count': 2, 'communication_skill_score': 7,
    'personal_groom_ing_score': 8, 'semester_cgpa': 9.1,
    'hackathon_participation_count': 2, 'other_competition_wins': 0,
    'avg_dm_conversations_per_week': 5, 'provides_academic_help': 1,
    'target_is_single': 1,  # Try changing this to 0 and see the result!
    'local_competition_score': 6
}
# The simulator function needs a slight modification for a key error in the user's input
# Correcting 'personal_groom_ing_score' to 'personal_grooming_score'
your_current_stats_v3['personal_grooming_score'] = your_current_stats_v3.pop(
    'personal_groom_ing_score')

predict_my_chances_v3(your_current_stats_v3, model_v3,
                      scaler_v3, feature_columns_v3)


# --- EDA for V3 ---
print("\n--- Generating V3 Feature Correlation Heatmap ---")
plt.figure(figsize=(20, 16))
sns.heatmap(df_v3.corr(), annot=True, cmap='coolwarm',
            fmt=".2f", annot_kws={"size": 10})
plt.title('V3 Advanced Feature Correlation Heatmap', fontsize=20)
plt.show()
