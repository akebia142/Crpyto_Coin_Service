#리팩토링중
# 데이터 수집
from konlpy.tag import Okt
import urllib.request
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_naver_npl_data():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                               filename="ratings_train.txt")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
                               filename="ratings_test.txt")

#pandas를 이용하여 데이터 로딩
def load_data():
    pd_train_data = pd.read_table('ratings_train.txt')
    pd_test_data = pd.read_table('ratings_test.txt')
    #print('2-1. 훈련용 리뷰 갯수 :', len(pd_train_data))
    #print('2-2. 테스트 리뷰 갯수 :', len(pd_test_data))
    #print('2-3. 데이터확인 ====')
    return pd_train_data,pd_test_data

def check_data(pd_train_data,pd_test_data):
    # 한글이외 데이터 제거
    reg_han = r"[^\sㄱ-ㅎ가-힣]"
    print("한글외 삭제 실행전", pd_train_data["document"][0])
    pd_train_data.replace(to_replace=reg_han, regex=True, inplace=True, value="")
    pd_test_data.replace(to_replace=reg_han, regex=True, inplace=True, value="")
    print("한글외 삭제 실행후", pd_train_data["document"][0])

    train_avalid_cnt= pd_train_data.isna().sum()
    test_avalid_cnt=pd_test_data.isna().sum()
    print("훈련용 전체 데이터수:",len(pd_train_data))
    print("테스트 전체 데이터수:",len(pd_test_data))
    print("유효하지 않는 훈련용데이터 수량", train_avalid_cnt)
    print("유효하지 않은 테스트데이터 수량", test_avalid_cnt)
    pd_train_data.dropna(axis=0, subset="document", inplace=True)
    pd_test_data.dropna(axis=0, subset="document", inplace=True)
    print("제거후 훈련용 전체 데이터수:", len(pd_train_data))
    print("제거후 훈련용 전체 데이터수:", len(pd_test_data))
    print("중복된 훈련용 데이터 수량",len(pd_train_data)-pd_train_data["document"].nunique()) #중복리뷰 3813
    print("중복된 테스트 데이터 수량", len(pd_test_data) - pd_test_data["document"].nunique())  # 중복리뷰 840
    pd_train_data.drop_duplicates(subset='document', inplace=True)
    pd_test_data.drop_duplicates(subset='document', inplace=True)
    print("중복제거후 훈련용 데이터 중복수량", len(pd_train_data) - pd_train_data["document"].nunique())
    print("중복제거후 테스트 데이터 중복수량", len(pd_test_data) - pd_test_data["document"].nunique())

#일단 불용어 제거 + 정답파일 분리
stopword = ["에서", "은", "는", "이", "가", "이다", "하다", "들", "좀", "걍", "도", "요", "에게", "나다", "데", "에", "의", "을", "를", "다",
                "한", "것", "내", "그", "나"]
def preprocess_data(pd_train_data, pd_test_data):
    okt = Okt()
    x_train = []
    for doc in (pd_train_data["document"][:5]):
        token_word = okt.morphs(doc, stem=True)  # 리턴값 : 단어 리스트
        x_train.append(" ".join([w for w in token_word if not w in stopword]))
    x_train = np.array(x_train)
    x_test = []
    for doc in tqdm(pd_test_data["document"][:5]):
        token_word = okt.morphs(doc, stem=True)  # 리턴값 : 단어 리스트
        x_test.append(" ".join([w for w in token_word if not w in stopword]))
    x_test = np.array(x_test)
    y_train= pd_train_data["label"].to_numpy()
    y_test= pd_test_data["label"].to_numpy()
    return(x_train,y_train),(x_test,y_test)

def preprocess_empty_remove(x_data,y_data):
    mask_index = (np.where((np.array([len(d) for d in x_data]) > 0)))
    x_data=x_data[mask_index]
    y_data=y_data[mask_index]
    print(x_data.shape)
    print(y_data.shape)
    return x_data,y_data


if __name__=="__main__":
    print("해당 모듈 실행")
    #get_naver_npl_data() #데이터 다운로드 한번하면 끝

    pd_train_data,pd_test_data=load_data()
    check_data(pd_train_data,pd_test_data)
    (x_train,y_train),(x_test,y_test)=preprocess_data(pd_train_data, pd_test_data)
    print(x_train[0])
    print(x_train.shape)
    print(y_train.shape)
