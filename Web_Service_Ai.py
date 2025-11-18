import base64

from flask import Flask,render_template,request,json,jsonify
app = Flask(__name__)
@app.route("/") #메인 인트로 페이지
def root():
    pass
@app.route("/crypto_coin") #코인 가격 예측 분석 페이지
def crpyto_coin():
    crpyto_coin_anal()
def crpyto_coin_anal()
    pass

app.run("127.0.0.1",4321,True)