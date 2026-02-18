<<<<<<< HEAD
# mnist-streamlit-app
Interactive Streamlit web app for classifying handwritten digits using a trained MNIST deep learning model.
=======
# MNIST Sifferigenkänning

Detta projekt demonstrerar utveckling av en maskininlärningsmodell för handskriven sifferigenkänning (MNIST) samt en interaktiv Streamlit-applikation för att testa modellen.

---

## Del 1: Modellutveckling

Se `mnist_model.ipynb` för:

- Exploratory Data Analysis (EDA)
- Jämförelse av flera modeller:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Classifier (SVC)
- GridSearch för hyperparameter-optimering
- Slutlig modell: **SVC med cirka 97.8% accuracy**

Den slutgiltiga modellen tränades på hela datasetet och sparades som:

```
mnist_svc_model.pkl
```

---

## Del 2: Streamlit-app

Applikationen låter användaren rita en siffra (0–9) direkt i webbläsaren och få en prediktion från den tränade modellen.

### Installera beroenden

```bash
pip install -r requirements.txt
```

### Starta appen

```bash
streamlit run mnist_app.py
```

---

## Funktioner

- Rita siffror direkt i webbläsaren
- Automatisk preprocessing (anpassad till träningspipeline)
- Prediktion av siffra 0–9
- Interaktivt stapeldiagram med sannolikheter
- Tydlig markering av modellens gissning
- Feedback-knappar: "Gissade jag rätt?"

## Syfte

Projektet visar en komplett ML-pipeline:

1. Dataanalys  
2. Modelljämförelse  
3. Hyperparameteroptimering  
4. Slutträning  
5. Deployment via Streamlit  
>>>>>>> 95bcb75 (Initial commit - MNIST Streamlit app)
