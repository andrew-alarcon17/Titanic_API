import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#API
@app.route('/api/status')
def status():
    return json.dumps({"status": "OK"})

@app.route('/api/predict', methods=['POST'])
def predict_v2():
    '''
    For rendering results on HTML GUI
    '''
    # parse request data as json
    app.logger.info("request data...")
    app.logger.info(request.json)
    values = request.get_json()
    app.logger.info(values)

    int_features = [int(values[k]) for k in values.keys()]
    app.logger.info(int_features)

    # store features as numpy array
    final_features = [np.array(int_features)]
    app.logger.info(final_features)

    # predit on testing data
    prediction = model.predict(final_features)
    app.logger.info(prediction)

    output = int(round(prediction[0], 2))

    # respond with model prediction
    return json.dumps({"status": "prediction OK", "Passenger Survived?": output})    

if __name__ == "__main__":
    app.run(debug=True)

