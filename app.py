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
from scipy.io import wavfile

app = Flask(__name__)
CORS(app)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
scaler = joblib.load("scaler.pkl")

@app.route('/', methods=['GET'])
def teste():
    return '<h1>Esse é o Teste de um modelo de Machine Learning para detecção de ronco</h1>'

@app.route('/predict', methods=['POST'])
def predict():
    print("Recebido áudio")
    file = request.files['audio']
    file.seek(0)  # reposiciona o ponteiro do arquivo para o início
    print(f"Arquivo {file} recebido")

    try:
        # Zera as predições e segmentos a cada requisição
        ronco_segments = []  # Zerar lista de segmentos
        predictions = []     # Zerar lista de predições
        prob = []            # Zerar lista de probabilidades
        audio_bytes = file.read()
        # audio_buffer = io.BytesIO(file.read())
        print("Vamos carregar o áudio com librosa")
        
        sr, audio = wavfile.read(io.BytesIO(audio_bytes))  # sr = sample rate, data = numpy array

        # audio, sr = librosa.load(audio_buffer, sr=16000)
        print("Áudio carregado com sucesso")
        segment_duration = sr  # 1 segundo = 16000 amostras
        num_segments = len(audio) // segment_duration
        print(f"Dividindo o áudio em {num_segments} segmentos de {segment_duration} amostras")
        print("Vamos processar os segmentos")
        for i in range(num_segments):
            start = i * segment_duration
            end = start + segment_duration
            segment = audio[start:end]

            if len(segment) < 512:
                continue

         # Converte para float32 normalizado
            segment = segment.astype(np.float32)
            segment = segment / np.max(np.abs(segment)) if np.max(np.abs(segment)) != 0 else segment

            # Gera espectrograma
            stft = librosa.stft(segment, n_fft=512, hop_length=256)
            spectrogram = np.abs(stft)
            db_spec = librosa.amplitude_to_db(spectrogram, ref=np.max)


            # Ajusta para 128 frames
            if db_spec.shape[1] < 128:
                pad_width = 128 - db_spec.shape[1]
                db_spec = np.pad(db_spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                db_spec = db_spec[:, :128]

            # Achata, normaliza e faz predição
            feat = db_spec.reshape(1, -1)
            feat_scaled = scaler.transform(feat)

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            interpreter.set_tensor(input_details[0]['index'], feat_scaled.astype(np.float32))
            interpreter.invoke()
            
            output = interpreter.get_tensor(output_details[0]['index'])
            prob.append(float(output[0][0]))

            prediction = int(output[0][0] > 0.5)
            predictions.append(prediction)

            if prediction == 1:
                segment_audio = audio[start:end]

                # Escreve em memória (sem salvar em disco)
                buffer = io.BytesIO()
                sf.write(buffer, segment_audio, sr, format='WAV')

                # Resetando o ponteiro do buffer para o início
                buffer.seek(0)
                # Lê os dados do buffer e codifica em base64
                audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

                ronco_segments.append({
                    'index': i,
                    'audio_base64': audio_base64
                })

        # Resumo da predição (opcional)
        percent_ronco = 100 * np.mean(predictions)
        resumo = "roncando" if percent_ronco > 50 else "normal"
        
        return jsonify({
            'prob': prob,
            'predictions': predictions,
            'percent_ronco': percent_ronco,
            'resumo': resumo,
            'ronco_segments': ronco_segments
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()