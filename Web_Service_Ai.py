import base64
from flask import Flask,render_template,request,json,jsonify
from ai_service.Crpyto_Coin_Service import input_request
app = Flask(__name__)
COIN_NAMES =["BTC","ETH","XRP"]
COIN_HAN =["비트코인","이더리움","리플"]
AI_PATH= "ai_service/"
def crypto_coin_anal(coinname,timegaps):
    res= input_request(coinname,timegaps)
    return res

@app.route("/") #메인 인트로 페이지
def root():
    return render_template("intro_index.html")
@app.route("/page/<pagename>")
def page_href(pagename):
    return render_template(f"{pagename}.html")
@app.route("/coin_name")
def out_coinname():
    coin_name_dict= {"eng_name":COIN_NAMES,"han_name":COIN_HAN}
    return jsonify(coin_name_dict)


@app.route("/user_data", methods=["POST"])
def user_data():
        user_datas = request.get_json()
        coinname = user_datas["coinname"]
        timegaps = int(user_datas["timegaps"])
        report = crypto_coin_anal(coinname, timegaps)

        result = {
            "status": "success",
            "coinname": coinname,
            "timegaps": timegaps,
            "report": report
        }
        print(report)
        return jsonify(result)

app.run("127.0.0.1",4321,True)