from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

def get_cleanded_data(form_data):
    gestation = float(form_data.get('gestation'))
    parity = float(form_data.get('parity'))
    age = float(form_data.get('age'))
    height = float(form_data.get('height'))
    weight = float(form_data.get('weight'))
    smoke = float(form_data.get('smoke'))

    cleaned_data = {
        "gestation": [gestation],
        "parity": [parity],
        "age": [age],
        "height": [height],
        "weight": [weight],
        "smoke": [smoke]
    }
    return cleaned_data


@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def get_predictions():
    # get data from user
    baby_data_form = request.form   # ✅ correct (no parentheses)
    baby_data_cleaned = get_cleanded_data(baby_data_form)

    # convert into data frame
    baby_df = pd.DataFrame(baby_data_cleaned)

    # ✅ ensure DataFrame is not empty
    if baby_df.empty:
        return render_template("index.html", prediction="Error: No input received")

    # load machine learning trained model
    with open("models/models.pkl", "rb") as obj:
        model = pickle.load(obj)

    # make predictions on user data
    prediction = model.predict(baby_df)
    prediction = round(float(prediction[0]), 2)

    # return response
    return render_template("index.html", prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
