# Sleep Health & Lifestyle Predictor

A web app that takes a person's lifestyle details and predicts their **sleep quality**, **stress level** and **sleep-disorder risk**, then gives personalised wellness recommendations. Three machine learning models are chained together and served through a **FastAPI** backend, with a simple **HTML** front end.

> This is a student machine learning project. It is not a medical tool and does not diagnose anything.

## The idea in plain words
Sleep, stress and health are connected. The app chains three predictions, where each result feeds the next one:

1. **Sleep quality** is predicted from the person's details.
2. **Stress level** is predicted from the same details plus the sleep-quality result.
3. **Sleep disorder** (None, Insomnia or Sleep Apnea) is predicted from the details plus sleep quality and stress.

## Inputs
Gender, age, occupation, sleep duration, physical activity (minutes per day), BMI category, blood pressure (e.g. 120/80), heart rate and daily steps.

## Models and results
Trained on an 80/20 train-test split (`random_state=42`).

| Step | Predicts | Model | Test result |
|---|---|---|---|
| 1 | Sleep quality (1 to 10) | Linear Regression | R² = 0.929 |
| 2 | Stress level (1 to 10) | Linear Regression | R² = 0.960 |
| 3 | Sleep disorder | Decision Tree (max depth 5) | Accuracy = 89.33% |

Categorical columns (gender, occupation, BMI category) are converted with a One-Hot Encoder, and blood pressure is split into systolic and diastolic values.

## Tech stack
Python, Pandas, Scikit-Learn, FastAPI, Uvicorn, Joblib, HTML, CSS, JavaScript

## Project structure
```
sleep-health-predictor/
├── train.ipynb                           # trains the models and saves them
├── main.py                               # FastAPI backend
├── index.html                            # web page front end
├── encoder.joblib                        # saved One-Hot Encoder
├── model1_sleep_quality.joblib           # saved model 1
├── model2_stress_level.joblib            # saved model 2
├── model3_sleep_disorder.joblib          # saved model 3
├── Sleep_health_and_lifestyle_dataset.csv.csv
└── requirements.txt
```

## How the pieces connect
1. `train.ipynb` reads the dataset, trains the three models and saves them as `.joblib` files.
2. `main.py` loads those `.joblib` files and exposes them at `POST /predict/chained`.
3. `index.html` sends the form data to that address and shows the response.

## How to run
1. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Re-train the models by running the first cell of `train.ipynb`. The saved model files are already included.
3. Start the backend from this folder:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
4. Open `index.html` in your browser, fill in the form and click **Run Complete Prediction Chain**.

You can also test the API at http://127.0.0.1:8000/docs.

## Dataset
Sleep Health and Lifestyle dataset (`Sleep_health_and_lifestyle_dataset.csv.csv`, 374 rows). 

## Limitations
- The dataset is small (374 rows), so results may not generalise to real people.
- Some occupations appear only a few times in the data, so predictions for them are less reliable.
- The model outputs are for learning purposes and are not medical advice.

## What I learned
- How to train and evaluate regression and classification models.
- How to chain models so one model's output becomes the next model's input.
- How to save models with Joblib and serve them through a FastAPI endpoint.
- How a front end talks to a backend.

## Team
This was a group project for a Machine Learning lab at COMSATS University Islamabad.
- Hammad Shafqat
- Abdullah Bin Rizwan
- Abdullah Shakeel (Me)
- All contributed to the Making of This Project, I focused on the Model training in `train.ipynb` (data preparation, encoding and training the models) and contributions to the front-end page `index.html`

## Next steps
- Compare more models, such as Random Forest, and use cross-validation.
- Add a confusion matrix for the sleep-disorder model.
- Deploy the app online.

## Power BI Dashboard
An interactive 2-page dashboard built on the same dataset.
- Page 1: overview and what drives sleep quality
- Page 2: who is at risk of a sleep disorder

![Overview](dashboard/page_1_dashb.png)
![Risk drill-down](dashboard/page_2_dashboard.png)

Open `dashboard/Sleep_Health_Dashboard.pbix` in Power BI Desktop (free, Windows) to explore it.
Small public dataset (374 rows, many repeated rows). Patterns show association, not cause. Not medical advice.

## Author
Abdullah Shakeel, Business Data Analytics student at COMSATS University Islamabad.
LinkedIn: https://www.linkedin.com/in/abdullahh-shakeel/


