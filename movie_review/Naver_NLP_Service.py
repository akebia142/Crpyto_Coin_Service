# 플라스크 서버와 연동되어 예측된 값을 전송

from movie_review.Naver_NLP_Predict import get_userData, vocab_processor, predict_userdata

def getPredict(userData):
    x_user = get_userData(userData)
    x_user = vocab_processor(x_user)
    res=predict_userdata(x_user)
    value=res[0].item()
    rat= value*100 if value>=0.5 else (1-value)*100
    context= f"{rat:.2f}% 확률로 긍정 리뷰입니다." if res[0]>=0.5 else f"{rat:.2f}% 확률로 부정리뷰입니다."
    return context

# if __name__=="__main__":
#     print(getPredict("아 영화 재밌다 bbb"))

