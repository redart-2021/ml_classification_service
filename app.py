import pandas as pd
from flask import Flask
from core.active_classificator import ActiveClassificator
from datetime import date
from flask import request, jsonify
import json


app = Flask(__name__)
act_class = ActiveClassificator(file_name="core/models/knn_model.model")

@app.route('/')
def servise_status():
    return jsonify({"Service status": "OK!", "used model": act_class.model_name})

@app.route('/predict', methods=['POST'])
def get_predict():
    req = dict(request.json)
    return {'user_type': str(act_class.predict([[req["right_fields"], req["wrong_fields"], req["user_stats"]]])[0])}

@app.route('/fit', methods=['POST'])
def fit(data):
    data = pd.json_normalize(json.loads(data))
    return act_class.fit(data[["lead_time", "stats"]].to_numpy())

if __name__ == '__main__':
    app.run()