# 定义策略网络
import numpy as np
import tensorflow as tf
import os


class Policy:
    '''
    策略网络
    输入：状态
    输出：一个高斯分布（正态分布） 均值和方差
    '''

    def __init__(self, obs_dim, act_dim):
        self.act_dim = act_dim
        self.obs_dim = obs_dim

    def build(self):
        inputs = tf.keras.layers.Input(shape=(self.obs_dim,))
        x = inputs
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x1 = tf.keras.layers.Dense(64, activation='tanh')(x)
        x2 = tf.keras.layers.Dense(64, activation='tanh')(x)
        plicy_mean = tf.keras.layers.Dense(self.act_dim, activation='linear')(x1)
        log_policy_std = tf.keras.layers.Dense(units=self.act_dim, activation='linear')(x2)

        # plicy_mean = plicy_mean*2
        # log_policy_std =  log_policy_std*10
        log_policy_std = tf.clip_by_value(log_policy_std, -20, 2)

        model = tf.keras.models.Model(inputs=inputs, outputs=[plicy_mean, log_policy_std])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4))
        # model.summary()
        return model


class Value:
    '''
    定义V网络-状态价值的近似，
    输入：状态
    输出：对状态的打分
    '''

    def __init__(self, obs_dim, act_dim,):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def build(self):
        inputs = tf.keras.layers.Input(shape=(self.obs_dim,))

        x = inputs
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(1)(x)
        outputs = x

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        # model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        return model


class Q:
    '''
    定义Q网络-动作价值函数的近似
    输入：动作，状态
    输出：打分
    '''

    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def build(self):
        inputs_act = tf.keras.layers.Input(shape=(self.act_dim,))
        inputs_obs = tf.keras.layers.Input(shape=(self.obs_dim,))

        inputs = tf.keras.layers.Concatenate()([inputs_obs, inputs_act])
        x = inputs
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(1)(x)
        outputs = x

        model = tf.keras.models.Model(inputs=[inputs_obs, inputs_act], outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        # model.summary()
        return model



from collections import deque
from tensorflow_probability.python.distributions import Normal
import random


class SAC:
    '''
    SAC算法连续版本
    '''
    def __init__(self, obs_dim, act_dim,act_scale=1,act_shifting=0,id = 0 ):


        replay_len = int(1e6)  # 经验的长度
        self.model_path = "model/"+ str(id) +"_model_{}.h5"

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_scale = act_scale
        self.act_shifting = act_shifting

        self.reward_scale = 5.0
        # 可训练参数
        self.log_aplha = tf.Variable(0.2, trainable=True, name="log EntropyTemperature")
        self.aplha = tf.Variable(0.2, trainable=True, name="EntropyTemperature")
        self.aplha.assign(tf.exp(self.log_aplha))
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)
        self.gamma = 0.99
        self.target_entropy = -np.prod(act_dim)
        self.alpha_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

        # 策略网络
        self.policy = Policy(obs_dim, act_dim).build()

        # V网络  弃用
        self.v = Value(obs_dim, act_dim).build()
        self.v_target = tf.keras.models.clone_model(self.v)

        # Q网络
        self.q_1 = Q(obs_dim, act_dim).build()
        self.q_2 = Q(obs_dim, act_dim).build()

        self.q_1_target = tf.keras.models.clone_model(self.q_1)
        self.q_2_target = tf.keras.models.clone_model(self.q_2)
        self.q_2.optimizers = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
        # 添加一个经验池子
        self.replay_buffer = deque(maxlen=replay_len)
        self.obs_normal_mean = 0
        self.obs_normal_std = 1

        self.reward_normal_mean = 0
        self.reward_normal_std = 1
        self.normal_flag = False

    def set_normal(self,reward_mean,reward_std,obs_mean=0,obs_std=1):
        self.obs_normal_mean = obs_mean
        self.obs_normal_std = obs_std

        self.reward_normal_mean = reward_mean
        self.reward_normal_std = reward_std


    def select_action(self, obs):
        # 选择动作
        tf.cast(obs,tf.float32)
        obs = tf.expand_dims(obs, axis=0)
        policy_mean, log_policy_std = self.policy(obs)
        act, logprob = self.process_action(policy_mean, log_policy_std)
        return act[0]

    def append_raplay(self, obs, act, reward, next_obs, done):
        # 添加到经验池
        reward = reward / self.reward_scale
        reward = (reward-self.reward_normal_mean)/self.reward_normal_std

        self.replay_buffer.append((obs, act,reward, next_obs, done))
    def sample_replay(self, batch_size):
        # 从经验池中采样
        mini_batch = random.sample(self.replay_buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)
        reward_batch = np.array(reward_batch).astype('float32')
        done_batch = np.array(done_batch).astype('float32')
        reward_batch = np.reshape(reward_batch, newshape=(-1, 1))
        done_batch = np.reshape(done_batch, newshape=(-1, 1))

        #对奖励和观察值归一化处理
        #obs_batch = (obs_batch-self.obs_normal_mean)/(self.obs_normal_std)
        #next_obs_batch = (next_obs_batch-self.obs_normal_mean)/(self.obs_normal_std)
        #reward_batch = (reward_batch-self.reward_normal_mean)/(self.reward_normal_std+1e-5)

        return np.array(obs_batch).astype('float32'), \
               np.array(action_batch).astype('float32'), reward_batch, \
               np.array(next_obs_batch).astype('float32'), done_batch

    def update_target_value(self, model, target_model, tau=0.005):
        # self.ema.apply(self.v.trainable_variables)
        # self.ema.apply(self.v_target.trainable_variables)
        weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        target_model.set_weights(target_weights)

    def process_action(self, policy_mean, log_policy_std):

        policy_std = tf.cast(tf.exp(log_policy_std), tf.float32)
        policy_mean = tf.cast(policy_mean, tf.float32)
        # 制造高斯分布
        gaussian_distribution = Normal(policy_mean, policy_std)
        # 从高斯分布采样
        gaussian_sampling = gaussian_distribution.sample()
        # noise = Normal(0,1)
        # z = noise.sample()
        sample_action = tf.tanh(gaussian_sampling)
        logprob = gaussian_distribution.log_prob(gaussian_sampling) - tf.math.log(
            1.0 - tf.pow(sample_action, 2) + 1e-6)
        logprob = tf.reduce_sum(logprob, axis=1, keepdims=True)
        return sample_action * self.act_scale+self.act_shifting, logprob

    @tf.function()
    def learn(self, obs, act, reward, next_obs, done):

        '''
        训练过程
        训练策略网络
        '''
        # 训练策略网络

        next_mean, next_log_std = self.policy(next_obs, training=False)
        # 预测动作
        next_act, next_logprob = self.process_action(next_mean, next_log_std)
        next_q = tf.stop_gradient(tf.math.minimum(self.q_1_target([next_obs, next_act], training=False),
                                                  self.q_2_target([next_obs, next_act], training=False)))
        with tf.GradientTape(persistent=True) as tape:
            policy_mean, log_policy_std = self.policy(obs, training=True)
            # 预测动作
            pre_act, logprob = self.process_action(policy_mean, log_policy_std)
            pre_Q = tf.math.minimum(self.q_1_target([obs, pre_act], training=False),
                                    self.q_2_target([obs, pre_act], training=False))
            # print("act",pre_act)
            policy_loss = tf.reduce_mean(self.aplha * logprob - pre_Q)

            alpha_loss = - tf.reduce_mean(self.log_aplha * (logprob + self.target_entropy))

            # v网络更新
            # pre_value = self.v(obs,training=True)
            # 此处出bug
            # value = pre_Q-self.aplha*logprob
            # value_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(pre_value,value))

            # Q网络没问题
            Q = tf.stop_gradient(reward + (next_q - next_logprob * self.aplha) * self.gamma * (1. - done))
            pre_Q1 = self.q_1([obs, act], training=True)
            pre_Q2 = self.q_2([obs, act], training=True)

            Q1_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(Q, pre_Q1))
            Q2_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(Q, pre_Q2))

        self.updata_model(tape, policy_loss, self.policy)
        alpha_grad = tape.gradient(alpha_loss, [self.log_aplha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_aplha]))

        # 更新V
        # self.updata_model(tape,value_loss,self.v)
        # 更新Q网络
        self.updata_model(tape, Q1_loss, self.q_1)
        self.updata_model(tape, Q2_loss, self.q_2)
        # 更新targetv

        # self.aplha.assign(tf.exp(self.log_aplha))
        return (
            Q1_loss,
            (Q2_loss),
            (policy_loss),
            # (value_loss),
            (alpha_loss),
            tf.exp(self.log_aplha),

        )

    def updata_model(self, tape, loss, model):
        # 更新神经网络用
        gradients = tape.gradient(loss, model.trainable_variables)

        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def load_weights(self):

        # 加载权重
        if not (os.path.exists(self.model_path.format("policy"))):
            return
        self.policy.load_weights(self.model_path.format("policy"))
        self.v.load_weights(self.model_path.format("v"))
        self.v_target.load_weights(self.model_path.format("v_target"))
        self.q_1.load_weights(self.model_path.format("q_1"))
        self.q_1_target = tf.keras.models.clone_model(self.q_1)
        self.q_2.load_weights(self.model_path.format("q_2"))
        self.q_2_target = tf.keras.models.clone_model(self.q_2)
    def save_weights(self):
        # 保存权重
        self.policy.save_weights(self.model_path.format("policy"))
        self.v.save_weights(self.model_path.format("v"))
        self.v_target.save_weights(self.model_path.format("v_target"))
        self.q_1.save_weights(self.model_path.format("q_1"))
        self.q_2.save_weights(self.model_path.format("q_2"))

    def train_step(self):
        if len(self.replay_buffer)<4000:
            return False
        b_obs, b_act, b_reward, b_next_obs, b_done = self.sample_replay(64)
        self.learn(b_obs, b_act, b_reward, b_next_obs, b_done)
        self.update_target_value(self.q_1, self.q_1_target)
        self.update_target_value(self.q_2, self.q_2_target)
        self.aplha.assign(tf.exp(self.log_aplha))

