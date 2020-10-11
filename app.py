import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    if(output == 0):
        pred = 'not authentic'
    else:
        pred = 'authentic'
    return render_template('index.html', prediction_text='The bank note is {}'.format(pred))

if __name__ == "__main__":
    app.run(debug=True)