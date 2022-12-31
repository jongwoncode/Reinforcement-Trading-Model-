import os
import time
import utils
import threading
import logging
import numpy as np
import tensorflow as tf

from network import LSTM_DNN_AC
from environment import Environment


# 멀티쓰레딩을 위한 글로벌 변수
global episode
episode= 0
num_episode = 2000
logger = logging.getLogger(utils.LOGGER_NAME)

# 액터러너 클래스 (쓰레드)
class Learner(threading.Thread):
    global_episode = 0

    def __init__(self, code, chart_data, training_data, initial_balance, 
                    min_trading_price, max_trading_price,
                    action_size, n_steps, chart_size, balance_size, 
                    global_model, optimizer, discount_factor, writer):
        threading.Thread.__init__(self)

        # A3CAgent 클래스에서 넘겨준 하이퍼 파라미터 설정
        self.code = code
        self.action_size = action_size
        self.n_steps = n_steps
        self.chart_size = chart_size
        self.balance_size = balance_size
        self.global_model = global_model
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.chart_states, self.balance_states, self.actions, self.rewards = [], [], [], []

        # 환경, 로컬신경망, 텐서보드 생성
        self.local_model = LSTM_DNN_AC(action_size, n_steps, chart_size, balance_size)
        self.env = Environment(chart_data, training_data, n_steps, initial_balance, min_trading_price, max_trading_price)
        self.writer = writer

        # 학습 정보를 기록할 변수
        self.avg_p_max = 0
        self.avg_loss = 0
        # k-타임스텝 값 설정(= batch)
        self.t_max = 15
        self.t = 0
        # 불필요한 행동을 줄여주기 위한 dictionary
        self.action_dict = {0:1, 1:2, 2:3, 3:3}

    # 텐서보드에 학습 정보를 기록
    def draw_tensorboard(self, trading_return, step, e):
        avg_p_max = self.avg_p_max / float(step)
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', trading_return, step=e)
            tf.summary.scalar('Average Max Prob/Episode', avg_p_max, step=e)
            tf.summary.scalar('Duration/Episode', step, step=e)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, c_observed, b_observed) :
        c_observed = tf.reshape(c_observed, shape= [-1, np.shape(c_observed)[0], np.shape(c_observed)[1]])
        b_observed = tf.reshape(b_observed, shape= [-1, np.shape(b_observed)[0]])
        policy = self.local_model([c_observed, b_observed])[0][0]
        policy = tf.nn.softmax(policy)
        action_index = np.random.choice(self.action_size, 1, p=policy.numpy())[0]
        return action_index, policy

    # 샘플을 저장
    def append_sample(self, c_history, b_history, action, reward) :
        self.chart_states.append(c_history)
        self.balance_states.append(b_history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # k-타임스텝의 prediction 계산
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0
        if not done:
            # value function
            c_last_state = tf.reshape(self.chart_states[-1], shape= [-1, np.shape(self.chart_states[-1])[0], np.shape(self.chart_states[-1])[1]])
            b_last_state = tf.reshape(self.balance_states[-1], shape= [-1, np.shape(self.balance_states[-1])[0]])
            running_add = self.local_model([c_last_state, b_last_state])[1][0].numpy()

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    # 저장된 샘플들로 A3C의 오류함수를 계산
    def compute_loss(self, done) :
        discounted_prediction = self.discounted_prediction(self.rewards, done)
        discounted_prediction = tf.convert_to_tensor(discounted_prediction, dtype=tf.float32)
        self.chart_states = tf.reshape(self.chart_states, shape= [-1, np.shape(self.chart_states)[1], np.shape(self.chart_states)[2]])
        self.balance_states = tf.reshape(self.balance_states, shape= [-1, np.shape(self.balance_states)[1]])      
        policy, values = self.local_model([self.chart_states, self.balance_states])

        # 가치 신경망 업데이트
        advantages = discounted_prediction - values
        critic_loss = 0.5 * tf.reduce_sum(tf.square(advantages))

        # 정책 신경망 업데이트
        action = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        policy_prob = tf.nn.softmax(policy)
        action_prob = tf.reduce_sum(action * policy_prob, axis=1, keepdims=True)
        cross_entropy = - tf.math.log(action_prob + 1e-10)
        actor_loss = tf.reduce_sum(cross_entropy * tf.stop_gradient(advantages))
        entropy = tf.reduce_sum(policy_prob * tf.math.log(policy_prob + 1e-10), axis=1)
        entropy = tf.reduce_sum(entropy)
        actor_loss += 0.01 * entropy

        total_loss = 0.5 * critic_loss + actor_loss

        return total_loss

    # 로컬신경망을 통해 그레이디언트를 계산하고, 글로벌 신경망을 계산된 그레이디언트로 업데이트
    def train_model(self, done):
        global_params = self.global_model.trainable_variables
        local_params = self.local_model.trainable_variables
        # (0) save calculated gradient information into 'tape' 
        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(done) 

        # (1) get local network gradient & clipping for stable learning
        grads = tape.gradient(total_loss, local_params)
        grads, _ = tf.clip_by_global_norm(grads, 40.0)
        # (2) update global network with local network gradient
        self.optimizer.apply_gradients(zip(grads, global_params))
        # (3) update local network with updated global network
        self.local_model.set_weights(self.global_model.get_weights())
        # (4) initialize sample
        self.chart_states, self.balance_states, self.actions, self.rewards = [], [], [], []

    def run(self):
        # 액터러너끼리 공유해야하는 글로벌 변수
        global episode
        step = 0
        while episode < num_episode :
            print('episode : ', episode)
            # 환경 초기화 및 초기 관찰값 확인
            self.env.reset()
            c_state, b_state, _, done = self.env.step(None, None)

            while not done :
                step += 1
                self.t += 1
                # 정책 확률에 따라 행동을 선택
                action, policy = self.get_action(c_state, b_state)
                # 선택한 행동으로 환경에서 한 타임스텝 진행
                c_next_state, b_next_state, reward, done = self.env.step(action, policy)
                # 정책확률의 최대값
                self.avg_p_max += np.amax(policy.numpy())
                # 샘플을 저장 : (s_t, a_t, r_t)
                self.append_sample(c_state, b_state, action, reward)
                c_state, b_state = c_next_state, b_next_state
                # batch 생성시 모델 훈련
                if self.t >= self.t_max or done :
                    # (done=True) : episode 종료or-50% 손실/(t>=t_max) : 최대 타임스텝 수에 도달(=batch size)
                    self.train_model(done)
                    self.t = 0

                # episode 종료시 학습 정보를 기록
                if done :
                    episode += 1
                    logger.debug(f'[{self.code}][Epoch {episode}/{num_episode}] '
                        f'#Buy:{self.env.num_long} #Sell:{self.env.num_short} #Hold:{self.env.num_hold} '
                        f'#Stocks:{self.env.num_stocks} PV:{self.env.portfolio_value:,.0f} '
                        f'profitloss:{self.env.profitloss:.6f}')
                    self.draw_tensorboard(self.env.profitloss, step, episode)
                    self.avg_p_max = 0
                    step = 0


