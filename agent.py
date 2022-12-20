import numpy as np
import utils

'''
Agent : 선물 거래를 염두한 행동 전략(long-short 배팅 가능)
    [Long-Short position]
        1. [훈련] Action에 의해 결정된 short position 물량이 현재 long 포지션의 물량을 넘으면 long 포지션을 모두 청산(close)하고 남은 물량을 open short
        2. [실전] Model을 수정.(+ 고립, 격리 반영)

    [state]
        1. 포지션_자금 비율      : postion_(t)/portfolio_value_(t)
        2. 손익                 : portfolio_value_(t)/portfolio_value_(0)
        3. 현재 포지션           : SHORT = 0, LONG = 1
        4. 평균 수익률
            * [LONG position]  close_price_(t) - mean_position_price_(t)
            * [SHORT position] mean_position_price_(t) - close_price_()
        
    [Slippage]
        1. 훈련 단계에서는 Slippage 적용.           (아직 미반영)
        2. 실제 거래 단계에서는 api를 통해 실제 balance를 조회하여 적용.
'''
class Agent :
    # agent state : [포지션-자금 비율, 손익, 평균 수익률, 현재 포지션]
    STATE_DIM = 4
    SHORT, LONG = 0, 1 
    # 매매 수수료(0.015%) 및 세금(0.25%)
    TRADING_CHARGE, TRADING_TAX  = 0.00015, 0.0025 
    # 행동 : short(0), hold(1), long(2)
    ACTION_SHORT, ACTION_HOLD, ACTION_LONG = 0, 1, 2 
    ACTIONS = [ACTION_SHORT, ACTION_HOLD, ACTION_LONG]    # action space
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self, environment, initial_balance, min_trading_price, max_trading_price) :
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment
        self.initial_balance = initial_balance  # 초기 자본금

        # 최소 단일 매매 금액, 최대 단일 매매 금액
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

 