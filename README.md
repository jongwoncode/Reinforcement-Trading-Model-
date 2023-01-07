# 강화학습을 활용한 트레이딩 봇

# 1. 프로젝트 정보
    
* 😀 프로젝트 구성원 : 이종원(개인프로젝트)
* 📆 프로젝트 기간  : 2022.12.15 ~ 2023.01.08
* 💻 사용 모델     : 강화학습 알고리즘(A3C), 신경망 모델(LSTM-DNN custom model)
* 🤖 주요 사용 기술  : python, tensorflow, keras, multi-processing

<br>

<br>

# 2. 목차
* **1. A3C 알고리즘 설명 및 선택 이유**
* **2. LSTMDNN 모델 설명 및 기본 모델(DNN)과 비교**
* **3. 파일 구조 및 설명**
* **4. 훈련 결과 확인**
* **5. cmd 입력**
* **6. 고찰**

<br>


<br>

# 3. A3C(Asynchronous Advantage Actor-Critic)
* **A3C(Asynchronous Advantage Actor-Critic) 모델 사용**
    - DQN의 장점을 차용하고 단점을 개선한 A3C모델 사용.
        + DQN 장점
            - 리플레이 메모리의 무작위 샘플 추출로 인한 샘플들의 상관관계 제거
        + DQN 단점
            - off-policy 방식
            - 리플레이의 메모리 데이터는 과거의 학습 정보
    - Multi Agent 방식 사용
        + Mutil Agent를 병렬 학습 하여 각 Agent마다 다른 상황에서 학습.
        + 주기적으로 local network를 global network로 업데이트. 
        + DQN의 replay memory의 랜덤 추출 기법의 아이디어를 적용.



<br>

<br>

<br>



<br>


<br>

# 5. cmd (argparser)

|입력인자|설명|type|default|
|--|--|--|--|
|$--name$|로그 파일의 이름|$string$|파일실행시간|
|$--code$|투자 종목 코드|$string$|'005380' (=현대자동차)|
|$--model$|신경망 모델 설정|$choice$ = ['LSTMDNN', 'DNN']|'LSTMDNN'|
|$--mode$|학습 모드|$choice$=['train', 'test', 'update', 'monkey']|'train'|
|$--start_date$|훈련 데이터 시작일|$string$|'20180601'|
|$--end_date$|훈련 데이터 마지막일|$string$|'20221220'|
|$--lr$|learning rate|$float$|0.0001|
|$--n_steps$|LSTMDNN Network의 n_steps|$int$|10|
|$--balance$|초기 잔고|$int$|100000000|

<br>

> ## main.py 내 입력 인자

```python
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=utils.get_time_str())
    parser.add_argument('--code', type=str, default='005380')
    parser.add_argument('--model', choices=['LSTMDNN', 'DNN'], default='LSTMDNN')
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'monkey'], default='train')
    parser.add_argument('--start_date', default='20180601')
    parser.add_argument('--end_date', default='20221220')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_steps', type=int, default=10)
    parser.add_argument('--balance', type=int, default=100000000)
    args = parser.parse_args()

```

<br>

> ## 예시
* 종목코드 '005380'을 n_step이 10인 LSTMDNN 신경망으로 훈련 시킴. learning rate는 0.0001. 훈련 데이터는 20200101 ~ 20221220
```
python main.py --code 005380 --model LSTMDNN --mode train --start_date 20200101 --lr 0.0001 -- n_steps 10
```

* 종목코드 '005380'을 n_step이 10인 기존에 훈련된 LSTMDNN 신경망으로 업데이트 시킴. learning rate는 0.00006, 업데이트 데이터는 20180601~20221220
```
python main.py --code 005380 --model LSTMDNN --mode update --start_date 20200101 --lr 0.00006
```

* 종목코드 '005380'을 DNN 신경망으로 훈련 시킴. learning rate는 0.0001, 업데이트 데이터는 20180601~20221220
```
python main.py --code 005380 --model DNN
```


<br>


<br>

# 6. 고찰 