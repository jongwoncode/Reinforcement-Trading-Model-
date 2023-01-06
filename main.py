import os
import sys
import logging
import argparse
import utils
import json
import data_preprocess 
from environment import Environment
from agent import A3CAgent

if __name__ == "__main__" :
    # argparse 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=utils.get_time_str())
    parser.add_argument('--code', type=str, default='005380')
    parser.add_argument('--model', choices=['LSTMDNN', 'DNN'], default='LSTMDNN')
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'monkey'], default='train')
    parser.add_argument('--start_date', default='20180601')
    parser.add_argument('--end_date', default='20221220')
    parser.add_argument('--lr', type=float, default=0.00004)
    parser.add_argument('--n_steps', type=int, default=10)
    parser.add_argument('--balance', type=int, default=100000000)
    args = parser.parse_args()

    # 학습기 파라미터 설정
    output_name = f'{args.mode}_{args.code}_{args.model}_{args.name}'
    # 출력 경로 생성
    output_path = os.path.join(utils.BASE_DIR, 'log')
    # 로그 파일 경로 생성
    log_path = os.path.join(output_path, f'{output_name}.log') 
    logger_name = utils.LOGGER_NAME 
    # 출력 경로 없다면 생성
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    # 기존에 로그 파일이 있다면 제거
    if os.path.exists(log_path):
        os.remove(log_path)

    # 파라미터 기록
    params = json.dumps(vars(args))
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)

    # log 기록을 위한 logger 설정
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)

    # preporcessing 파라미터 설정
    preprocess_params = {'code' : args.code, 'model' : args.model, 'start_date' : args.start_date, 
                            'end_date' : args.end_date, 'n_steps' :args.n_steps}
    # 데이터 전처리&학습 데이터 생성
    chart_data, training_data = data_preprocess.load_data(**preprocess_params)

    # network에 학습으로 입력될 feature들의 개수 설정(chart_size=24, balance_size=4)
    if args.model == 'LSTMDNN' :
        chart_size, balance_size = training_data.shape[2], 4
    elif args.model == 'DNN' :
        chart_size, balance_size = training_data.shape[1], 4

    # 최소/최대 단일 매매 금액 설정
    min_trading_price, max_trading_price = 500000, 3000000
    

    # Agent 학습 파라미터 설정
    agent_params = {'code' :args.code, 'model' : args.model, 'n_steps' : args.n_steps, 'chart_size' : chart_size, 'balance_size' :balance_size, 
                        'action_size' : 3, 'chart_data' : chart_data, 'training_data' :training_data, 
                        'initial_balance' : args.balance, 'min_trading_price' : min_trading_price, 
                        'max_trading_price' : max_trading_price, 'lr' : args.lr}


    # 모델 학습 모드에 따른 실행
    # 랜덤 선택
    if args.mode == 'monkey' :
        agent_params.update({'reuse_model' : False})
        global_agent = A3CAgent(**agent_params)
        global_agent.monkey()

    # 모델 훈련
    elif args.mode == 'train' :
        agent_params.update({'reuse_model' : False})
        global_agent = A3CAgent(**agent_params)
        global_agent.train()

    # 모델 업데이트
    elif args.mode == 'update' :
        agent_params.update({'reuse_model' : True})
        global_agent = A3CAgent(**agent_params)
        global_agent.train()

    # 모델 테스트
    elif args.mode == 'test' :
        agent_params.update({'reuse_model' : True})
        global_agent = A3CAgent(**agent_params)
        global_agent.test()


