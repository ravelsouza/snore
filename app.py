from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
from flask_cors import CORS
import io
import base64
import soundfile as sf

app = Flask(__name__)
CORS(app)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
scaler = joblib.load("scaler.pkl")

@app.route('/', methods=['GET'])
def teste():
    return '<h1>Esse é o Teste de um modelo de Machine Learning para detecção de ronco</h1>'

if __name__ == '__main__':
    app.run()