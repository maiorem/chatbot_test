from flask import Flask, render_template, request
from flask import flash
import pandas as pd
import training_bot
import dialogue

app = Flask(__name__)
app.secret_key = 'MIEW'

@app.route("/")
@app.route("/index")
def index_page():
    return render_template('index.html')

@app.route('/training', methods=['POST'])
def training():
    chatbot_data = pd.read_excel(request.files['file'])
    training_bot.main_training(chatbot_data)

    flash("훈련이 완료되었습니다.")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    question=request.form['question']
    input_seq = dialogue.make_predict_input(question)
    sentence = dialogue.generate_text(input_seq)
    print(sentence)
    return render_template('index.html')


if __name__ ==  "__main__":
    app.run()
