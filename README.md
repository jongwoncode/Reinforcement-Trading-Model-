# 강화학습을 활용한 트레이딩 봇

# 1. 프로젝트 정보
    
* 😀 프로젝트 구성원 : 이종원(개인프로젝트)
* 📆 프로젝트 기간  : 2022.12.15 ~ 2023.01.08
* 💻 사용 모델     : 강화학습 알고리즘(A3C), 신경망 모델(LSTM-DNN custom model)
* 🤖 주요 사용 기술  : python, tensorflow, keras, multi-processing
* 🤑 프로젝트 설명 : [종원이의 Velog](https://velog.io/@leejong)


<br>

# cmd (argparser)

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

