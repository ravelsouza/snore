from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/teste', methods=['POST'])
def teste():
    return '<h1>Esse é o Teste de um modelo de Machine Learning para detecção de ronco</h1>'

if __name__ == '__main__':
    app.run()