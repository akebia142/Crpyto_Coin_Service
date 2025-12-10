#저장된 모델의 예측값 출력
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from konlpy.tag import Okt
import re
import tensorflow as tf

PATH=r"movie_review/" #웹서비스 파일에서 실행
#PATH=r"./" #현재 파일에서 실행
#모델생성
model = None
#가중치 셋팅
#print(model.get_weights()[:5])
MAX_TOKENS=0
MAX_LEN=0
VOCAB=None

if os.path.exists(f"{PATH}config/config"):
    with open(f"{PATH}config/config","rb") as fp:
        configs = pickle.load(fp)
    MAX_TOKENS=configs["max_tokens"]
    MAX_LEN=configs["max_len"]
    VOCAB=configs["vocab"]
    #print(configs["max_tokens"]-1)
model=tf.keras.models.load_model(f"{PATH}config/nlp_model.keras",compile=True)
    #레이어 이름도 동일해야합니다.
    #print(model.get_weights()[:5])
    #사전설정
tv=tf.keras.layers.TextVectorization(
    output_mode='int',
    output_sequence_length=MAX_LEN,
)
tv.set_vocabulary(VOCAB)
def get_userData(user_data):# 이 영화는 너무 재밌어
    # 정규식 전환, 불용어처리/형태소분류, 숫자변환(Tokenizer-vocab)
    reg_han = r"[^\sㄱ-ㅎ가-힣]"
    user_data = re.sub(reg_han,"",user_data)
    #user_data.replace(to_replace=reg_han, regex=True, inplace=True, value="")
    if not user_data :
        print("좀 더 명확한 입력을 해주세요")
    stopword = ["에서", "은", "는", "이", "가", "이다", "하다", "들", "좀", "걍", "도", "요",
                "흠", "에게", "나다", "데", "있다", "해도", "에", "의", "을", "를", "다", "한",
                "것", "내", "그", "나"]
    # 나중에 단어 출현 횟수에 따라 의미없는 단어는 추가하여 다시 제거를 하는게 좋다.
    #print("한글 형태소 분리 실행")
    okt = Okt()
    x_user = []
    token_word = okt.morphs(user_data, stem=True)  # 리턴값 : 단어 리스트
    x_user.append(" ".join([w for w in token_word if not w in stopword]))
    x_user = np.array(x_user)
    return x_user

def vocab_processor(x_user):
    global tv
    return tv(x_user)
def predict_userdata(x_user):
    global model
    return model.predict(x_user)

if __name__=="__main__":

    x_user = get_userData("영화 재밌네 다음에는 밤새서 와야겠다 푹 자게")
    print(x_user)
    print(vocab_processor(x_user))
    x_user2= vocab_processor(x_user)
    print(predict_userdata(x_user2))
