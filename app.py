from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__, template_folder='templates')

# 加载训练好的模型
model_path = os.path.join(os.path.expanduser("~/Desktop/my_project"), 'best_model.pkl')
with open(model_path, 'rb') as f:
    best_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 获取用户在网页上输入的特征值
    UCR = float(request.form.get('UCR'))
    SBP = float(request.form.get('SBP'))
    Respiratory_rate = float(request.form.get('Respiratory_rate'))
    Temperature = float(request.form.get('Temperature'))
    SpO2 = float(request.form.get('SpO2'))
    Age = float(request.form.get('Age'))
    Aniongap = float(request.form.get('Aniongap'))
    Potassium = float(request.form.get('Potassium'))
    PT = float(request.form.get('PT'))
    PTT = float(request.form.get('PTT'))
    ALT = float(request.form.get('ALT'))
    ALP = float(request.form.get('ALP'))
    AST = float(request.form.get('AST'))
    Liver_disease = float(request.form.get('Liver_disease'))

    # 将特征值组成一个列表，用于模型预测
    features = [UCR, SBP, Respiratory_rate, Temperature, SpO2, Age, Aniongap, Potassium, PT, PTT, ALT, ALP, AST, Liver_disease]

    # 进行预测
    prediction = best_model.predict([features])[0]

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)