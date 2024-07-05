from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

model = joblib.load(r'C:\Users\gowth\OneDrive\Desktop\ProgrammingProjects\CodSoft\spam_sms_detection\model\trained_model.pkl')  # Replace with your model file path
vectorizer = joblib.load(r'C:\Users\gowth\OneDrive\Desktop\ProgrammingProjects\CodSoft\spam_sms_detection\model\count_vectorizer.pkl')  # Replace with your vectorizer file path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sms = request.form['sms']
        sms_vectorized = vectorizer.transform([sms])
        prediction = model.predict(sms_vectorized)
        sms_label = 'spam' if prediction[0] == 1 else 'ham'
        return render_template('result.html', sms_label=sms_label, sms_text=sms)

if __name__ == '__main__':
    app.run(debug=True)
