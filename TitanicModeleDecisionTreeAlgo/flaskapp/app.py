from flask import Flask, request, render_template
from sklearn.tree import DecisionTreeClassifier
import joblib
import pandas as pd

app = Flask(__name__)

# Charger les modèles
model_gini = joblib.load('titanic_dt_gini.pkl')
model_entropy = joblib.load('titanic_dt_entropy.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    message = "" 
    if request.method == "POST":
        algorithme = request.form.get("algorithme")
        lesex = request.form.get("lesex")
        Survived = request.form.get("Survived")
        SibSp = request.form.get("SibSp")
        age = request.form.get("age")
       
        if algorithme:
            if SibSp and age and lesex and Survived:
                new_data = {
                    'Survived': int(Survived),
                    'Sex': int(lesex),
                    'Age': float(age),
                    'SibSp': int(SibSp)
                }
                # Création d'un DataFrame à partir des nouvelles données
                X_new = pd.DataFrame([new_data])
                if algorithme == "decisiontreeentropy":
                    prediction = model_entropy.predict(X_new)
                    return render_template('index.html', message=f"La prédiction est {prediction[0]}. C'est selon l'algorithme DT-Entropy.")
                else:
                    prediction = model_gini.predict(X_new)
                    return render_template('index.html', message=f"La prédiction est {prediction[0]}. C'est selon l'algorithme DT-Gini.")
            else:
                return render_template('index.html', message="Erreur: Veuillez fournir tous les paramètres.")
        else:
            return render_template('index.html', message="Erreur: Veuillez sélectionner un algorithme.")
    else:
        return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
