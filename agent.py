import os
import tensorflow as tf
import numpy as np

from environment import MnistEnvironment, Environment
from replay_memory import ReplayMemory
from ddpg_model import DDPG
from origin_model.mnist_solver import Network

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
                    self.replay.add(state, action, reward, next_state, terminal)
    
                    if len(self.replay.memory) >= self.batch_size:
                        self.ddpg.update_target_network()
                        self.ddpg.train_network()
    
                    score += reward
                    state = np.reshape(next_state, [1, self.state_size])
    
                    if terminal:
                        scores.append(score)
                        episodes.append(e)
                        if (i+1)%10 == 0:
                            print('epoch', e+1, 'iter:', f'{i+1:05d}', ' score:', f'{score:.03f}', ' last 10 mean score', f'{np.mean(scores[-min(10, len(scores)):]):.03f}', f'sequence: {self.env.sequence}')
                        if (i+1)%500 == 0:
                            self.ENV.render_worker(os.path.join(self.render_dir, f'{(i+1):05d}.png'))
                        if (i+1)%1000 == 0:
                            self.save()

        pass

    def play(self):
        cor_before_lst, cor_after_lst = [], []
        for idx in range(self.test_size): 
            state = self.ENV.new_episode(idx, phase='test')
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

