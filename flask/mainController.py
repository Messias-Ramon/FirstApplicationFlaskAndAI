from flask import Flask, request, render_template 
import random

lista = ["A", "B", "C"]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('tela.html')

@app.route('/submit', methods=['POST'])
def submit():
    return render_template('tela.html', resp = lista[random.randint(0,2)])  # Retorna o nome com a mensagem

if __name__ == '__main__':
    app.run()