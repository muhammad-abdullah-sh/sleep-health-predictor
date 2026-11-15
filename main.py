from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved OneHotEncoder and trained models
encoder = joblib.load("encoder.joblib")
m1 = joblib.load("model1_sleep_quality.joblib")
m2 = joblib.load("model2_stress_level.joblib")
m3 = joblib.load("model3_sleep_disorder.joblib")

class HealthInput(BaseModel):
    gender: str
    age: int
    occupation: str
    sleep_duration: float
    physical_activity: int
    bmi_category: str
    blood_pressure: str
    heart_rate: int
    daily_steps: int

# NEW: keeps displayed scores on a 1-10 scale
def clamp_score(value: float, low: float = 1.0, high: float = 10.0) -> float:
    return max(low, min(high, value))

def get_sleep_suggestion(score: float) -> str:
    if score >= 7.5:
        return "Optimal sleep quality. Keep maintaining your steady sleep-wake schedule."
    elif score >= 5.5:
        return "Moderate sleep quality. Aim for 30-45 minutes more sleep and avoid evening caffeine."
    return "Poor sleep quality. Minimize screen time before bed and maintain strict bedtime hours."

def get_stress_suggestion(score: float) -> str:
    if score <= 4.0:
        return "Low stress levels. Your routine and recovery habits are well balanced."
    elif score <= 6.5:
        return "Moderate stress. Incorporate short active breaks and light evening stretching."
    return "High stress. Practice breathing exercises and ensure adequate restorative rest."

def get_disorder_advice(disorder: str) -> str:
    if disorder == "None":
        return "No sleep disorder detected. Maintain your active and balanced routine."
    elif disorder == "Insomnia":
        return "Indicators align with Insomnia. Maintain fixed wake times and avoid tossing in bed awake."
    return "Indicators align with Sleep Apnea. Consult a doctor for an overnight sleep study (polysomnography)."

# NEW: personalised recommendations based on the user's inputs and the predictions
def get_recommendations(data, quality: float, stress: float, disorder: str,
                        sys_bp: int, dia_bp: int) -> list:
    recs = []

    # Sleep duration
    if data.sleep_duration < 6:
        recs.append("You are sleeping under 6 hours. Aim for 7-8 hours by moving your bedtime earlier in 15-minute steps.")
    elif data.sleep_duration < 7:
        recs.append("Try to extend your sleep to at least 7 hours a night for better recovery and mood.")

    # Physical activity
    if data.physical_activity < 30:
        recs.append("Your activity level is low. You should exercise more: start with 20-30 minutes of brisk walking, cycling or light jogging most days.")
    elif data.physical_activity < 60:
        recs.append("Good start on activity. Work towards about 150 minutes of moderate exercise per week, plus 2 days of strength training.")
    else:
        recs.append("Your activity level is great. Keep it up, but avoid intense workouts within 2-3 hours of bedtime.")

    # Daily steps
    if data.daily_steps < 5000:
        recs.append("Your daily steps are low. Increase gradually by 1,000 steps every week towards 8,000-10,000 steps a day.")
    elif data.daily_steps < 8000:
        recs.append("You are close to a healthy step count. A short walk after meals can get you to 8,000+ steps.")

    # Stress and work-life balance
    if stress >= 7:
        recs.append("Your stress is high. Consider reducing your working hours or workload, delegating tasks, and setting a firm cut-off time for work each day.")
        recs.append("Take a 5-10 minute break every 90 minutes of work, and practice deep breathing, meditation or yoga daily.")
        recs.append("If stress stays high for weeks, talk to a trusted person or a mental health professional.")
    elif stress > 4:
        recs.append("Schedule regular breaks, protect your free time, and make room for hobbies or social time to keep stress in check.")

    # Sleep quality habits
    if quality < 5.5:
        recs.append("Improve your sleep environment: keep the room dark, quiet and cool, and stop screens at least 1 hour before bed.")
        recs.append("Avoid caffeine after early afternoon and avoid heavy meals or alcohol close to bedtime.")
    elif quality < 7.5:
        recs.append("Keep a consistent sleep and wake time every day, including weekends, to improve sleep quality further.")

    # BMI
    bmi = data.bmi_category.strip().lower()
    if bmi in ("overweight", "obese"):
        recs.append("Consider a balanced diet with more vegetables, fruit and whole grains, and gradual weight management. Excess weight can worsen sleep quality and breathing during sleep.")

    # Heart rate
    if data.heart_rate > 100:
        recs.append("Your resting heart rate is high. Limit caffeine, manage stress and consult a doctor if it stays this high.")
    elif data.heart_rate > 80:
        recs.append("Your heart rate is slightly elevated. Regular cardio exercise and reducing caffeine can help lower it over time.")

    # Blood pressure
    if sys_bp >= 140 or dia_bp >= 90:
        recs.append("Your blood pressure reading is high. Reduce salt intake, limit caffeine and alcohol, and please consult a doctor.")
    elif sys_bp >= 130 or dia_bp >= 85:
        recs.append("Your blood pressure is slightly elevated. Cut back on salt, stay active and monitor it regularly.")

    # Predicted sleep disorder
    if disorder == "Insomnia":
        recs.append("For possible insomnia: keep fixed wake times, use the bed only for sleep, and try a relaxing wind-down routine. Consider talking to a doctor about CBT-I.")
    elif disorder not in ("None", "Insomnia"):
        recs.append("For possible sleep apnea: sleep on your side, avoid alcohol before bed, and see a doctor about a sleep study.")

    if not recs:
        recs.append("Your habits look healthy. Keep up your current routine.")

    return recs

@app.get("/")
def home():
    return {"message": "Server is up and running!"}

@app.post("/predict/chained")
def predict_chained(data: HealthInput):
    sys_bp, dia_bp = map(int, data.blood_pressure.split('/'))

    # Transform categorical variables using the loaded OneHotEncoder
    cat_df = pd.DataFrame([{
        'Gender': data.gender,
        'Occupation': data.occupation,
        'BMI Category': data.bmi_category
    }])
    encoded_cats = encoder.transform(cat_df)
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out())

    # Build numeric features
    num_df = pd.DataFrame([{
        'Age': data.age,
        'Sleep Duration': data.sleep_duration,
        'Physical Activity Level': data.physical_activity,
        'Heart Rate': data.heart_rate,
        'Daily Steps': data.daily_steps,
        'Systolic_BP': sys_bp,
        'Diastolic_BP': dia_bp
    }])

    X_base = pd.concat([num_df, encoded_df], axis=1)

    # 1. Predict Quality of Sleep
    pred_quality = float(m1.predict(X_base)[0])

    # 2. Predict Stress Level (Base + Quality)
    X2 = X_base.copy()
    X2['Quality of Sleep'] = pred_quality
    pred_stress = float(m2.predict(X2)[0])

    # 3. Predict Sleep Disorder (Base + Quality + Stress)
    X3 = X2.copy()
    X3['Stress Level'] = pred_stress
    pred_disorder = str(m3.predict(X3)[0])

    # NEW: cap displayed scores to the 1-10 scale (the models still run on raw values)
    shown_quality = clamp_score(pred_quality)
    shown_stress = clamp_score(pred_stress)

    return {
        "quality_of_sleep": round(shown_quality, 2),
        "quality_suggestion": get_sleep_suggestion(shown_quality),
        "stress_level": round(shown_stress, 2),
        "stress_suggestion": get_stress_suggestion(shown_stress),
        "sleep_disorder": pred_disorder,
        "disorder_advice": get_disorder_advice(pred_disorder),
        # NEW fields
        "scale": "Sleep quality and stress level are shown on a 1-10 scale",
        "recommendations": get_recommendations(
            data, shown_quality, shown_stress, pred_disorder, sys_bp, dia_bp
        ),
        "disclaimer": "These suggestions are general wellness guidance, not a medical diagnosis. Consult a healthcare professional for medical concerns."
    }
