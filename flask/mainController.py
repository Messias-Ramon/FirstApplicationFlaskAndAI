from flask import Flask, request, render_template 
import random


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('tela.html')

@app.route('/submit', methods=['POST'])
def submit():
    return render_template('tela.html', sub_component = "window", probability = "28%")  # Retorna o nome com a mensagem

if __name__ == '__main__':
    app.run()