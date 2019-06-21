# import cv2
import os
import sys
import gym
import random
import numpy as np
import argparse
import tensorflow as tf
from collections import deque
from copy import copy
import sys
import time
from tensorflow.examples.tutorials.mnist import input_data

from origin_model.mnist_solver import Network
import util

class MnistEnvironment(object):
    def __init__(self, model):
        self.model = model
        self.mc = 15
        self.threshold = 5e-3
        self._max_episode_steps = 15
        
        self.state_size = 784
        self.action_size = 2
        self.a_bound = np.array([10,2.5])
        
        self.data_load()
    
    def data_load(self):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
        
        self.train_images = mnist.train.images.reshape([-1,28,28,1])
        self.train_labels = mnist.train.labels
        self.test_images = mnist.test.images.reshape([-1,28,28,1])[:200]
        self.test_labels = mnist.test.labels[:200]
            
    def reset(self, idx, phase='train'):
        self.phase = phase
        if self.phase == 'train':
            self.img = self.train_images[idx] # 28*28*1
            self.img = util.random_degrade(self.img)
            self.label = self.train_labels[idx]
        else: # self.phase == 'test'
            self.img = self.test_images[idx]
            self.img = util.random_degrade(self.img)
            self.label = self.test_labels[idx]

        # initialize
        self.sequence = 0
        self.batch_imgs = [self.img] # save the rotated images 
        self.del_angles = [0] # save the rotated angle sequentially
        prob_set = util.all_prob(self.model, np.expand_dims(self.img, axis=0), self.mc)
        self.uncs = [util.get_mutual_informations(prob_set)[0]] # save the uncertainty
        self.label_hats = [prob_set.sum(axis=0).argmax(axis=1)[0]] # save predicted label

        return self.img.flatten()
    
    def step(self, rotate_angle, sharpen_radius):
        # sequence
        self.sequence += 1

        # next_state
        next_img = util.np_rotate(self.img, sum(self.del_angles))
#         next_img = util.np_sharpen(next_img, sharpen_radius)
        
        # reward
        prob_set = util.all_prob(self.model, np.expand_dims(next_img, axis=0), self.mc)
        unc_after = util.get_mutual_informations(prob_set)[0]
        unc_before = self.uncs[-1]

        reward_after = np.clip(-np.log(unc_after), a_min=None, a_max=-np.log(self.threshold))
        reward_before = np.clip(-np.log(unc_before), a_min=None, a_max=-np.log(self.threshold))
        reward = reward_after - reward_before
        
        # save the values
        self.del_angles.append(rotate_angle)
        self.uncs.append(unc_after)
        self.label_hats.append(prob_set.sum(axis=0).argmax(axis=1)[0])
        self.batch_imgs.append(next_img)
        
        # terminal
        if self.phase == 'train':
            if (unc_after < self.threshold and self.label_hats[-1] == self.label) \
               or self.sequence >= self._max_episode_steps:
                terminal = True
            else:
                terminal = False
        else: # self.phase == 'test'
            if unc_after < self.threshold or self.sequence >= self._max_episode_steps:
                terminal = True
            else:
                terminal = False
        
        return next_img.flatten(), reward, terminal, 0
        
    def render(self, fname):
        self.batch_imgs = np.stack(self.batch_imgs)
        img_width = self.batch_imgs.shape[2]
        
        self.batch_imgs = util.make_grid(self.batch_imgs, len(self.batch_imgs), 2)
        print(self.uncs,'\n')
        tick_labels = [f'{angle:.02f}\n{unc:.04f}\n{label_hat}'
                       for (angle, unc, label_hat) 
                       in zip(self.del_angles, self.uncs, self.label_hats)]
        util.save_batch_fig(fname, self.batch_imgs, img_width, tick_labels)

    def compare_accuracy(self):
        return (self.label_hats[0] == self.label, self.label_hats[-1] == self.label)


class Environment(object):
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        pass

    def new_episode(self, idx, phase='train'):
        state = self.env.reset(idx, phase)
        return state
        pass

    def act(self, action):
        next_state, reward, terminal, _ = self.env.step(*action)
        return next_state, reward, terminal
        pass

    def render_worker(self, fname):
        self.env.render(fname)
        pass

    def compare_accuracy(self):
        return self.env.compare_accuracy()
        pass


class ReplayMemory(object):
    def __init__(self, state_size, batch_size):
        self.memory = deque(maxlen=10000)
        self.state_size = state_size
        self.batch_size = batch_size
        pass

    def add(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
        pass

    def mini_batch(self):
        mini_batch = random.sample(self.memory, self.batch_size)  # memory에서 random하게 sample

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, terminals = [], [], []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            next_states[i] = mini_batch[i][3]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            terminals.append(mini_batch[i][4])
        return states, actions, rewards, next_states, terminals
        pass


class DDPG(object):
    def __init__(self, state_size,  action_size, sess, learning_rate_actor, learning_rate_critic,
                 replay, discount_factor, a_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess
        self.lr_actor = learning_rate_actor
        self.lr_critic = learning_rate_critic
        self.replay = replay
        self.discount_factor = discount_factor
        self.action_limit = a_bound

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.target = tf.placeholder(tf.float32, [None, 1])

        self.actor = self.build_actor('actor_eval', True)
        self.actor_target = self.build_actor('actor_target', False)
        self.critic = self.build_critic('critic_eval', True, self.actor)
        self.critic_target = self.build_critic('critic_target', False, self.actor_target)

        self.actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_eval')
        self.actor_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
        self.critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_eval')
        self.critic_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')

        self.replace = [tf.assign(t, (1 - 0.01) * t + 0.01 * e) 
                        for t, e in zip(self.actor_target_vars + self.critic_target_vars, self.actor_vars + self.critic_vars)]

        self.train_actor = self.actor_optimizer()
        self.train_critic = self.critic_optimizer()
        pass

    def build_actor(self, scope, trainable):
        actor_hidden_size = 30
        with tf.variable_scope(scope):
            hidden1 = tf.layers.dense(self.state, actor_hidden_size, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(hidden1, self.action_size, activation=tf.nn.tanh, name='a', trainable=trainable)
            return a * self.action_limit + np.array([0., 2.5])

    def build_critic(self, scope, trainable, a):
        with tf.variable_scope(scope):
            critic_hidden_size =30
            hidden1 = tf.layers.dense(self.state, critic_hidden_size, name='s1', trainable=trainable) \
                      + tf.layers.dense(a, critic_hidden_size, name='a1', trainable=trainable) \
                      + tf.get_variable('b1', [1, critic_hidden_size], trainable=trainable)
            hidden1 = tf.nn.relu(hidden1)
            return tf.layers.dense(hidden1, 1, trainable=trainable)

    def actor_optimizer(self):
        loss = tf.reduce_mean(self.critic)
        train_op = tf.train.AdamOptimizer(-self.lr_actor).minimize(loss, var_list=self.actor_vars)

        return train_op
        pass

    def critic_optimizer(self):
        loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.critic)
        #loss = tf.reduce_mean(tf.square(self.target - self.critic))
        train_op = tf.train.AdamOptimizer(self.lr_critic).minimize(loss, var_list=self.critic_vars)
        return train_op
        pass

    def train_network(self):
        states, actions, rewards, next_states, terminals = self.replay.mini_batch()

        next_target_q = self.sess.run(self.critic_target, feed_dict={self.state: next_states})

        target = []
        for i in range(self.replay.batch_size):
            if terminals[i]:
                target.append(rewards[i])
            else:
                target.append(rewards[i] + self.discount_factor * next_target_q[i])
        target = np.reshape(target, [self.replay.batch_size, 1])

        self.sess.run(self.train_actor, feed_dict={self.state: states})
        self.sess.run(self.train_critic, feed_dict={self.state: states, self.target: target, self.actor: actions})
        pass

    def update_target_network(self):
        self.sess.run(self.replace)
        pass


class Agent(object):
    def __init__(self, args, sess):
        # CartPole 환경
        self.sess = sess
        self.model = Network(sess, phase='train') # mnist accurcacy model
        self.env = MnistEnvironment(self.model) 
        self.state_size = self.env.state_size
        self.action_size = self.env.action_size
        self.a_bound = self.env.a_bound
        self.train_size = len(self.env.train_images)
        self.test_size = len(self.env.test_images)
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.discount_factor = args.discount_factor
        self.epochs = args.epochs
        self.ENV = Environment(self.env, self.state_size, self.action_size)
        self.replay = ReplayMemory(self.state_size, self.batch_size)
        self.ddpg = DDPG(self.state_size, self.action_size, self.sess, self.learning_rate[0], self.learning_rate[1], 
                         self.replay, self.discount_factor, self.a_bound)

        self.save_dir = args.save_dir
        self.render_dir = args.render_dir
        self.play_dir = args.play_dir

        # initialize
        sess.run(tf.global_variables_initializer())  # tensorflow graph가 다 만들어지고 난 후에 해야됨

        # load pre-trained mnist model
        self.env.model.checkpoint_load()
        
        self.saver = tf.train.Saver()
        self.epsilon = 1
        self.explore = 2e4
        pass

    '''
    def select_action(self, state):
        return np.clip(
            np.random.normal(self.sess.run(self.ddpg.actor, {self.ddpg.state: state})[0], self.action_variance), -2,
            2)
        pass
    '''

    def ou_function(self, mu, theta, sigma):
        x = np.ones(self.action_size) * mu
        dx = theta * (mu - x) + sigma * np.random.randn(self.action_size)
        return x + dx

    def noise_select_action(self, state):
        action = self.sess.run(self.ddpg.actor, {self.ddpg.state: state})[0]
        noise = self.epsilon * self.ou_function(0, 0.15, 0.25)
        return action + noise

    def select_action(self, state):
        return self.sess.run(self.ddpg.actor, {self.ddpg.state: state})[0]

    def train(self):
        scores, episodes = [], []
        for e in range(self.epochs):
            for i, idx in enumerate(np.random.permutation(self.train_size)):
                terminal = False
                score = 0
                state = self.ENV.new_episode(idx)
                state = np.reshape(state, [1, self.state_size])

                while not terminal:
                    action = self.noise_select_action(state)
                    next_state, reward, terminal = self.ENV.act(action)
                    state = state[0]
                    self.replay.add(state, action, reward / 10, next_state, terminal)
    
                    if len(self.replay.memory) >= self.batch_size:
                        self.ddpg.update_target_network()
                        self.ddpg.train_network()
    
                    score += reward
                    state = np.reshape(next_state, [1, self.state_size])
    
                    if terminal:
                        scores.append(score)
                        episodes.append(e)
                        if (i+1)%10 == 0:
                            print('epoch', e+1, 'iter:', f'{i+1:05d}', ' score:', score, ' last 10 mean score', np.mean(scores[-min(10, len(scores)):]), f'sequence: {self.env.sequence}')
                        if (i+1)%500 == 0:
                            self.ENV.render_worker(os.path.join(self.render_dir, f'{(i+1):05d}.png'))
                        if (i+1)%1000 == 0:
                            self.save()

        pass

    def play(self):
        cor_before_lst, cor_after_lst = [], []
        for idx in range(self.test_size): 
            state = self.ENV.new_episode(idx)
            state = np.reshape(state, [1, self.state_size])
    
            terminal = False
            score = 0
            while not terminal:
                action = self.select_action(state)
                next_state, reward, terminal = self.ENV.act(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                score += reward
                state = next_state
#                 time.sleep(0.02)
                if terminal:
                    (cor_before, cor_after) = self.ENV.compare_accuracy()
                    cor_before_lst.append(cor_before)
                    cor_after_lst.append(cor_after)

                    self.ENV.render_worker(os.path.join(self.play_dir, f'{(idx+1):04d}.png'))
                    print(f'{(idx+1):04d} image score: {score}\n')
        print('====== NUMBER OF CORRECTION =======')
        print(f'before: {np.sum(cor_before_lst)}, after: {np.sum(cor_after_lst)}')
    pass

    def save(self):
        checkpoint_dir = os.path.join(self.save_dir, 'ckpt')
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))

    def load(self):
        checkpoint_dir = os.path.join(self.save_dir, 'ckpt')
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))


if __name__ == "__main__":
    print(sys.executable)
    # parameter 저장하는 parser
    parser = argparse.ArgumentParser(description="Pendulum")
    parser.add_argument('--gpu_number', default='0', type=str)
    parser.add_argument('--learning_rate', default=[0.002, 0.001], type=list)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--epochs', default=1, type=float)
    parser.add_argument('--save_dir', default='save2', type=str)
    parser.add_argument('--render_dir', default='render_train', type=str)
    parser.add_argument('--play_dir', default='render_test', type=str)
    sys.argv = ['-f']
    args = parser.parse_args()

    args.render_dir = os.path.join(args.save_dir, args.render_dir)
    args.play_dir = os.path.join(args.save_dir, args.play_dir)
    if not os.path.exists(args.render_dir):
        os.makedirs(args.render_dir)
    if not os.path.exists(args.play_dir):
        os.makedirs(args.play_dir)

    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
    config.log_device_placement = False
    config.gpu_options.allow_growth = True

    # 학습 or 테스트
    with tf.Session(config=config) as sess:
        agent = Agent(args, sess)

        agent.train()
        agent.save()
#         agent.load()


