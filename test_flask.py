# Using flask to make an api
# import necessary libraries and functions
from dataclasses import dataclass
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import tensorflow as tf

scaler = StandardScaler()
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)

model.load_weights("model.h5")

print("Loaded model from disk successfully")

fs = 20
frame_size = fs*10 # 200
hop_size = fs*2 # 40
model = model

# creating a Flask app
app = Flask(__name__)
  
# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/', methods = ['GET', 'POST'])
def home():
    if(request.method == 'GET'):
        data = "hello world"
        return jsonify({'data': data})
  
  
# A simple function to calculate the square of a number
# the number to be squared is sent in the URL when we use GET
# on the terminal type: curl http://127.0.0.1:5000 / home / 10
# this returns 100 (square of 10)
@app.route('/inference/<string:x>/<string:y>/<string:z>', methods = ['GET'])
def disp(x, y, z):
    x_inp = str(x) 
    y_inp = str(y)
    z_inp = str(z)
    x_data = []
    for xt in x_inp.split(","):
        x_data.append(float(xt))
    y_data = []
    for yt in y_inp.split(","):
        y_data.append(float(yt))
    z_data = []
    for zt in z_inp.split(","):
        z_data.append(float(zt))
    test_data = pd.DataFrame(list(zip(x_data, y_data, z_data)), columns =['x', 'y', 'z'])
    X_test_scaled = scaler.fit_transform(test_data)
    X_test_scaled = pd.DataFrame(data = X_test_scaled, columns = ['x', 'y', 'z'])
    
    def get_frames_inference(df, frame_size, hop_size):
        N_FEATURES = 3
        frames = []
        for i in range(0, len(df) - frame_size, hop_size):
            x = df['x'].values[i: i + frame_size]
            y = df['y'].values[i: i + frame_size]
            z = df['z'].values[i: i + frame_size]
            frames.append([x, y, z])
        # Bring the segments into a better shape
        frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
        return frames
    
    X_test_data = get_frames_inference(X_test_scaled, frame_size, hop_size)
    X_test_data = X_test_data.reshape(X_test_data.shape[0],200,3,1)

    pred = model.predict(X_test_data)
    data = np.array(pred)
    y_pred = np.argmax(pred, axis=-1)

    if y_pred[0] == 1:
        op = "Drunk"
    else:
        op = "Not Drunk"

    return jsonify({'output': op})
  
    # return jsonify({'data': len(x) + len(y) + len(z)})
  
# driver function
if __name__ == '__main__':
  
    app.run(debug = True)
