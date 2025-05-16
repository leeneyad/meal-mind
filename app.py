from flask import Flask, render_template, request
import pandas as pd
import joblib

# تحميل النموذج والبيانات
model = joblib.load('random_forest_model.pkl')
meals_df = pd.read_csv('meals_data.csv')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    insulin = float(request.form['insulin'])

    meals = meals_df.copy()
    meals['age'] = age
    meals['bmi'] = bmi
    meals['insulin'] = insulin

    X = meals[['age', 'bmi', 'insulin', 'protein_(g)', 'carbohydrates_(g)', 'fat_(g)', 'calories_(kcal)']]
    meals['predicted_glucose'] = model.predict(X)

    top3 = meals.sort_values('predicted_glucose').head(3)['food_item'].tolist()
    
    return render_template('index.html',results=top3)

if __name__== '__main__':
    app.run(debug=True)