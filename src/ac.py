#
# command examples:
#  python3 src/ac.py
#  python3 src/ac.py --no_recurrent --ke=0.1
#

import os
import argparse
import time
import gym
import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import time
from utils import Plot

class policy_ac(Model):

    def __init__(self, num_actions, input_dim=(1, 8), hidden_dim=128, recurrent=False, k_entropy = 0.01):
        super(policy_ac, self).__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.recurrent = recurrent
        self.k_clipping = 0.2
        self.k_entropy = k_entropy
        self.eps = 1e-10

        self.features = self.make_features()
        self.actor = self.make_actor()
        self.critic = self.make_critic()
        self.build((1,)+input_dim)
        print(self.summary())
        print(self.features.summary())
        print(self.actor.summary())
        print(self.critic.summary())


    def make_features(self):
        m = tf.keras.Sequential(name="featues")
        if self.recurrent:
            input_shape=(1, )+self.input_dim
            print("--->>> recurrent ", input_shape)
            m.add(GRU(units=self.hidden_dim, return_sequences=False, batch_input_shape=input_shape, activation="tanh", stateful=True))
        else:
            m.add(Flatten())
            m.add(Dense(self.hidden_dim, activation="relu", dtype=tf.float32))
        return m

    def make_actor(self):
        m = tf.keras.Sequential(name="actor")
        m.add(Dense(self.num_actions, activation="softmax"))
        return m

    def make_critic(self):
        m = tf.keras.Sequential(name="critic")
        m.add(Dense(1, activation=None))
        return m

    def entropy(self, prob):
        entropy = -K.mean((prob * K.log(prob + self.eps)))
        return entropy

    def actor_loss(self, discounted_rewards, critic_values, action_probs, actions):

        # This code selects the probabilities for those actions that were actually played.
        actions_onehot = tf.one_hot(actions, self.num_actions)
        prob = tf.reduce_sum(action_probs * actions_onehot, axis=-1)
        log_prob = tf.math.log(prob)
        actor_loss = log_prob * (discounted_rewards - critic_values)

        #
        entropy_loss = self.k_entropy * self.entropy(action_probs)
        total_loss = -actor_loss - entropy_loss

        # return loss as sum of batch lossess
        total_loss = tf.reduce_sum(total_loss)
        return total_loss

    def critic_loss(self, critic_value, discounted_reward):
        # loss as sum of batch RMS lossess
        loss = tf.reduce_sum((critic_value - discounted_reward) ** 2)
        return loss

    def call(self, states):
        feat = self.features(states)
        probs = self.actor(feat)
        values = tf.squeeze(self.critic(feat), axis=-1)
        return probs, values

    def reset_states(self):
        if self.recurrent:
            self.features.reset_states()

class agent_ac:


    def __init__(self, env_name="LunarLander-v2", lr=0.0001, k_entropy=0.001, render_period=100, plot=True, recurrent=False):
        self.env = gym.make(env_name)
        print(f"env_name={env_name} max_episode_steps={self.env._max_episode_steps} reward_threshold={self.env.spec.reward_threshold}")
        self.num_actions = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.env._max_episode_steps = self.env.spec.reward_threshold * 3 // 2
        self.num_episodes = 10000
        self.gamma = 0.99  # Discount factor for past rewards
        self.lr = lr
        self.beta_1 = 0.9
        self.eps = 1e-10
        self.optimizer  = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1, epsilon=self.eps)
        self.episode_rewards = np.zeros(0)
        self.entropies = np.zeros(0)
        self.policy = policy_ac(num_actions=self.num_actions, input_dim=(1, self.state_size[0]), recurrent=recurrent, k_entropy=k_entropy)
        self.replay_epochs = 10 # update policy for 10 epochs
        self.best_average_reward = 0
        self.lr_update_threshold = 0
        self.render_period = render_period
        self.plot = plot


    def sample(self, probs):
        logits = np.log(probs)
        noise = np.random.uniform(size=logits.shape)
        return np.argmax(logits - np.log(-np.log(noise)))


    def action(self, state):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        action_probs, critic_value = self.policy(tf_state)
        action = self.sample(action_probs)
        #action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))
        #print(f"state={state} action_probs={action_probs} critic_value={critic_value} action={action}")
        return action, action_probs, critic_value


    def update_average(self, values, value):
        nsamples = 30
        values = np.append(values, value)
        if len(values) > nsamples:
            values = values[-nsamples:]
        avg = np.mean(values)
        return avg, values


    def run(self):

        plot = Plot(num_subplots=2, subplot_title=["average score", "entropy"])

        if self.render_period > 0:
            r = self.render()
            print(f"reward={r}")

        for episode in range(self.num_episodes):

            states_history = []
            actions_history = []
            critic_values_history = []
            action_probs_history = []
            rewards_history = []
            episode_steps = 0
            done = False

            t_start = time.time()
            state = self.env.reset()
            self.policy.reset_states()
            episode_reward = 0
            episode += 1

            # play and record states, actions, action probabilities, rewards
            with tf.GradientTape() as tape:
                while not done:
                    # state shape is (number_of_timesteps, state_size) with number_of_timesteps=1
                    state = state[np.newaxis, ...]
                    action, action_probs, critic_value = self.action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    states_history.append(state)
                    actions_history.append(action)
                    rewards_history.append(reward)

                    # using index 0 because of output batch size of 1
                    action_probs_history.append(action_probs[0])
                    critic_values_history.append(critic_value[0])

                    state = next_state
                    episode_reward += reward
                    episode_steps += 1

                # train policy with recorded values
                actor_loss, critic_loss = self.replay(actions_history, action_probs_history, critic_values_history, rewards_history)
                loss = actor_loss + critic_loss
                grads = tape.gradient(loss, self.policy.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

            # update average entropy and reward
            entropy = self.entropy(np.array(action_probs_history))
            average_entropy, self.entropies = self.update_average(self.entropies, entropy)
            average_reward, self.episode_rewards = self.update_average(self.episode_rewards, episode_reward)

            # plot averages
            if self.plot:
                plot.update(value=average_reward, idx=0)
                plot.update(value=average_entropy, idx=1)

            # update learning rate
            if (average_reward > self.best_average_reward):
                self.best_average_reward = average_reward
                lr = self.optimizer._decayed_lr('float32').numpy()
                if (average_reward > self.lr_update_threshold):
                    self.lr_update_threshold = average_reward * 1.5
                    lr = lr * 0.99
                    K.set_value(self.optimizer.learning_rate, lr)
                    print(f"new max average={average_reward}/{self.env.spec.reward_threshold}; learning rate={lr:.7f}")

            # render test
            if self.render_period > 0 and (episode % self.render_period) == 0:
                r = self.render()
                print(f"test episode={episode} reward={r}")

            # print episode summary
            t_end = time.time()
            if episode % 10 == 0:
                print("episode: {}/{} fps={:.2f} reward last/avg/max_avg: {:.2f}/{:.2f}/{:.2f} loss actor={:.5f} critic={:.5f} entropy={:.5f} sequence len={:d}"\
                      .format(episode,  self.num_episodes, len(rewards_history) / (t_end - t_start),\
                              episode_reward, average_reward, self.best_average_reward,
                              actor_loss, critic_loss, average_entropy, len(rewards_history)))

            # exit if solved
            if self.best_average_reward > self.env.spec.reward_threshold:
                print(f"completed in {episode} episodes;  average score={self.best_average_reward} threshold={self.env.spec.reward_threshold}")
                break


    def replay(self, actions_history, action_probs_history, critic_values_history, rewards_history):
        # actions_history, action_probs_history, critic_values_history, rewards_history are python lists

        # compute discounted rewards
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
        returns = returns.tolist()

        # compute losses
        returns = tf.stack(returns)
        critic_values_history = tf.stack(critic_values_history)
        action_probs_history = tf.stack(action_probs_history)
        actions_history = tf.stack(actions_history)
        actor_loss = self.policy.actor_loss(returns, critic_values_history, action_probs_history, actions_history)
        critic_loss = self.policy.critic_loss(critic_values_history, returns)

        #print(f"actor_loss={actor_loss} actor_loss2={actor_loss2}")
        #print(f"critic_loss={critic_loss} critic_loss2={critic_loss2}")

        return actor_loss, critic_loss


    def entropy(self, probs):
        entropy = np.mean(-np.sum(probs * np.log(probs + self.eps), axis=-1), axis=-1)
        return entropy


    def render(self):
        state = self.env.reset()
        self.policy.reset_states()
        done = False
        total_reward = 0
        while not done:
            state = state[np.newaxis, ...]
            action, _, _ = self.action(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.env.render()
        return total_reward


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
    parser.add_argument('--no_recurrent', action='store_true', default=False, help='on: use Dense; off: use RNN for feature extraction')
    parser.add_argument('--no_render', action='store_true', default=False, help='render test')
    parser.add_argument('--no_plot', action='store_true', default=False, help='disable average score plot ')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--env', type=str, default="CartPole-v0", help='gym environment name')
    parser.add_argument('--ke', type=float, default=0.01, help='entopy coefficient')
    args = parser.parse_args()
    args.plot = not args.no_plot
    args.render_period = 0 if args.no_render else 100
    args.recurrent  = not args.no_recurrent
    return args


def main():
    args = get_args()
    print(f"ARGS:")
    for arg in vars(args):
        print(f"{arg}={getattr(args, arg)}")

    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) <=0:
            print("no cuda devices")
            exit(0)
        print(f'GPUs {gpus}')
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    agent = agent_ac(env_name=args.env, lr=args.lr, render_period=args.render_period, plot=args.plot, k_entropy=args.ke, recurrent=args.recurrent)
    agent.run()

    agent.env._max_episode_steps = 1000
    while True:
        reward = agent.render()
        print(f"reward={reward}")

if __name__ == "__main__":
    main()
