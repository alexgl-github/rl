import sys
import os
import random
import copy
import time
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from utils import Plot
import argparse

class ActorCritic(Model):
    def __init__(self, input_shape, action_space, k_entropy=0.01):
        super(ActorCritic, self).__init__()
        self.action_space = action_space

        self.k_clipping = 0.2
        self.k_entropy = k_entropy
        self.eps = 1e-10

        self.actor = tf.keras.Sequential()
        self.actor.add(Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.actor.add(Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.actor.add(Dense(128, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.actor.add(Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.actor.add(Dense(self.action_space, activation="softmax"))

        self.critic = tf.keras.Sequential(name="critic")
        self.critic.add(Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.critic.add(Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.critic.add(Dense(128, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.critic.add(Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
        self.critic.add(Dense(1, activation=None))


    def loss_actor(self, advantages, probs, actions, y_pred):

        actions_onehot = tf.one_hot(actions, self.action_space, dtype=tf.float64)
        prob = tf.reduce_sum(actions_onehot * y_pred, axis=-1)
        old_prob = tf.reduce_sum(actions_onehot * probs, axis=-1)
        prob = K.clip(prob, self.eps, 1.0)
        old_prob = K.clip(old_prob, self.eps, 1.0)
        ratio = K.exp(K.log(prob) - K.log(old_prob))
        surr1 = ratio * advantages
        surr2 = K.clip(ratio, min_value = 1.0 - self.k_clipping, max_value = 1.0 + self.k_clipping) * advantages
        actor_loss = -K.mean(K.minimum(surr1, surr2))

        entropy = -(y_pred * K.log(y_pred + self.eps))
        entropy = self.k_entropy * K.mean(entropy)

        loss = actor_loss - entropy
        return loss, entropy

    def loss_critic(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2)
        return value_loss

    def call(self, states):
        probs = self.actor(states)
        values = tf.squeeze(self.critic(states), axis=-1)
        return probs, values


class Agent:
    def __init__(self, env_name, k_entropy=0.1, lr=1e-3, num_episodes=10000, render_period=0, plot=True):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env._max_episode_steps = self.env.spec.reward_threshold * 3 // 2
        print(f"env_name={env_name} max_episode_steps={self.env._max_episode_steps} reward_threshold={self.env.spec.reward_threshold}")
        self.num_actions = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.num_episodes = num_episodes
        self.lr = lr
        self.num_replay_epochs = 5
        self.optimizer_actor  = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.optimizer_critic  = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.ac = ActorCritic(input_shape=self.state_size, action_space=self.num_actions, k_entropy=k_entropy)
        self.render_period = 0
        self.gae_gamma = 0.99
        self.gae_lambda = 0.9
        self.eps = 1e-10
        self.scores = np.zeros(0)
        self.entropies = np.zeros(0)
        self.plot = plot

    def update_average(self, values, value):
        values = np.append(values, value)
        if len(values) > 40:
            values = values[-40:]
        avg = np.mean(values)
        return avg, values

    def act(self, state):
        prob, value = self.ac(state)
        prob = prob[0] # batch of 1
        value = value[0] # batch of 1
        assert np.abs(np.sum(prob) - 1.0) <= self.eps, "prob probabilities don't sum up to 1.0 prob={}".format(prob)
        action = np.random.choice(self.num_actions, p=prob)
        return action, prob, value

    def get_gaes(self, rewards, dones, values, next_values):
        q_targets = rewards + self.gae_gamma * next_values * (1-dones)
        deltas = q_targets - values
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * self.gae_gamma * self.gae_lambda * gaes[t + 1]

        target = gaes + values
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return gaes, target

    def run(self):

        plot = Plot(num_subplots=2, subplot_title=["average score", "entropy"])
        average_score = 0.0
        average_entropy = 0.0

        for episode in range(1, self.num_episodes+1):

            state = self.env.reset()
            state = state[np.newaxis, ...]
            done, score = False, 0
            states, actions, rewards, probs, values, dones = [], [], [], [], [], []

            while not done:
                if self.render_period > 0 and episode % self.render_period == 0:
                    self.env.render()
                action, prob, value = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                probs.append(prob)
                values.append(value)
                state = next_state[np.newaxis, ...]
                score += reward

                if done:
                    loss_actor, loss_critic, entropy = self.replay(states, actions, rewards, probs, values, dones, next_state)
                    average_score, self.scores = self.update_average(self.scores, score)
                    average_entropy, self.entropies = self.update_average(self.entropies, entropy)
                    print("episode: {}/{}, score: {}, average: {:.2f} loss actor={:.4f} critic={:.4f} entropy={:.4f} num states={}"
                          .format(episode, self.num_episodes, score, average_score, loss_actor, loss_critic, average_entropy, len(states)))

                    if self.plot:
                        plot.update(value=average_score, idx=0)
                        plot.update(value=average_entropy, idx=1)

            if average_score >= self.env.spec.reward_threshold:
                print(f"solved in {episode} episodes")
                break


    def replay(self, states, actions, rewards, probs, critic_values, dones, next_state):
        states = np.vstack(states)
        next_state = next_state[np.newaxis, ...]
        next_states = np.append(states[1:], next_state, axis=0)
        actions = np.array(actions)
        probs = np.vstack(probs)
        critic_values = np.array(critic_values)
        next_values = critic_values[1:]
        _, next_value = self.ac(next_state)
        next_values = np.append(next_values, next_value)
        dones = 1*np.array(dones)
        advantages, target = self.get_gaes(rewards, dones, critic_values, next_values)

        actor_loss = 0
        critic_loss = 0
        entropy = 0
        for _ in range(self.num_replay_epochs):
            with tf.GradientTape() as tape:
                y_pred, values_pred = self.ac(states)
                a_loss, ent = self.ac.loss_actor(advantages,
                                                 probs,
                                                 actions,
                                                 y_pred)
                actor_loss += a_loss
                entropy += ent

                c_loss = self.ac.loss_critic(target,
                                             values_pred)
                critic_loss += c_loss

                grad = tape.gradient(a_loss + c_loss, self.ac.trainable_variables)

            self.optimizer_actor.apply_gradients(zip(grad, self.ac.trainable_variables))

        actor_loss = actor_loss / self.num_replay_epochs
        critic_loss = critic_loss / self.num_replay_epochs
        return actor_loss, critic_loss, entropy


    def test(self):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done = False
        score = 0
        while not done:
            if self.render_period >= 0:
                self.env.render()
            probs, _ = self.ac(state)
            action = np.argmax(probs[0])
            state, reward, done, _ = self.env.step(action)
            state = np.reshape(state, [1, self.state_size[0]])
            score += reward
            if done:
                print("test score: {}".format(score))
                break

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
    parser.add_argument('--no_recurrent', action='store_true', default=False, help='on: use Dense; off: use RNN for feature extraction')
    parser.add_argument('--no_render', action='store_true', default=False, help='render test')
    parser.add_argument('--no_plot', action='store_true', default=False, help='disable average score plot ')
    parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes')
    parser.add_argument('--env', type=str, default="CartPole-v0", help='gym environment name')
    parser.add_argument('--ke', type=float, default=0.05, help='entopy coefficient')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    args.plot = not args.no_plot
    args.render_period = 0 if args.no_render else 100
    args.recurrent  = not args.no_recurrent
    return args


def main():
    args = get_args()
    print(f"PPO args:")
    for arg in vars(args):
        print(f"{arg}={getattr(args, arg)}")

    agent = Agent(env_name=args.env, k_entropy=args.ke, lr=args.lr, num_episodes=args.num_episodes, render_period=args.render_period, plot=args.plot)
    agent.run()
    while True:
        agent.test()

if __name__ == "__main__":
    main()
