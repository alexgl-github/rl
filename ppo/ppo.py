import os
import argparse
import time
import gym
import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, GRU
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import time


class actor_critic(Model):

    def __init__(self, num_actions, input_dim=(1, 8), hidden_dim=200, recurrent=False):
        super(actor_critic, self).__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.recurrent = recurrent
        self.features = self.make_features()
        self.actor = self.make_actor()
        self.critic = self.make_critic()
        self.clipping = 0.2
        self.k_entropy = 0.001
        self.eps = 1e-10

    def make_features(self):
        m = tf.keras.Sequential()
        if self.recurrent:
            m.add(GRU(units=self.hidden_dim, return_sequences=False, input_shape=self.input_dim, activation="tanh"))
        else:
            m.add(Dense(512, activation="tanh"))
            m.add(Dense(512, activation="tanh"))
        return m


    def make_actor(self):
        m = tf.keras.Sequential()
        m.add(Dense(256, activation="tanh"))
        m.add(Dense(128, activation="tanh"))
        m.add(Dense(self.num_actions, activation="softmax"))
        return m


    def make_critic(self):
        m = tf.keras.Sequential()
        # make critic stronger then actor
        m.add(Dense(512, activation="tanh"))
        m.add(Dense(256, activation="tanh"))
        m.add(Dense(128, activation="tanh"))
        m.add(Dense(1, activation=None))
        return m


    def loss_actor(self, advantages, action_probs, actions_onehot, y_pred):
        prob = actions_onehot * y_pred
        prob = K.clip(prob, self.eps, 1.0)
        old_prob = actions_onehot * action_probs
        old_prob = K.clip(old_prob, self.eps, 1.0)
        ratio = K.exp(K.log(prob) - K.log(old_prob))
        surr_loss1 = ratio * advantages
        surr_loss2 = K.clip(ratio, min_value = 1 - self.clipping, max_value = 1 + self.clipping) * advantages
        actor_loss = -K.mean(K.minimum(surr_loss1, surr_loss2))
        entropy = -(y_pred * K.log(y_pred + self.eps))
        entropy = self.k_entropy * K.mean(entropy)
        loss_total = actor_loss - entropy
        return loss_total


    def loss_critic(self, values, y_true, y_pred):
        LOSS_CLIPPING = 0.2
        clipped_value_loss = values + K.clip(y_pred - values, -self.clipping, self.clipping)
        v_loss1 = (y_true - clipped_value_loss) ** 2
        v_loss2 = (y_true - y_pred) ** 2
        value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
        return value_loss


    def act(self, x):
        feat = self.features(x)
        prob = self.actor(feat)
        return prob


    def critique(self, x):
        feat = self.features(x)
        v = self.critic(feat)
        return v


class Plot:

    def __init__(self, title=""):
        plt.ion()
        self.fig = plt.figure()
        plt.title(title)
        self.ax = self.fig.add_subplot(111)
        self.y = np.zeros((0,))
        self.x =  np.zeros((0,))
        self.line, = self.ax.plot(self.x, self.y, 'r-')


    def update(self, value):
        self.y = np.append(self.y, value)
        self.x = [i for i in range(0, len(self.y))]
        self.ax.relim()
        self.ax.autoscale_view()
        self.line.set_ydata(self.y)
        self.line.set_xdata(self.x)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class Agent:

    def __init__(self, env_name = "LunarLander-v2", lr=0.0001, play=False, recurrent=False, plot=True):
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.num_episodes = 10000
        self.lr = lr
        self.eps = 1e-5
        self.optimizer  = tf.keras.optimizers.Adam(learning_rate=self.lr, epsilon=self.eps)
        self.scores = np.zeros(0)
        self.ac = actor_critic(num_actions=self.action_size, recurrent=recurrent)
        self.replay_epochs = 10 # update policy for 10 epochs
        self.max_score_avg = 0
        self.play = play
        self.plot = plot


    def action(self, state):
        if self.ac.recurrent:
            action_probs = self.ac.act(np.reshape(state, (1, 1, state.shape[-1])))
        else:
            action_probs = self.ac.act(np.reshape(state, (1, state.shape[-1])))
        action = np.random.choice(self.action_size, p=np.squeeze(action_probs))
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, action_probs

    def update_score(self, score):
        self.scores = np.append(self.scores, score)
        if len(self.scores) > 50:
            self.scores = self.scores[-50:]
        avg = np.mean(self.scores)
        return avg


    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)


    def replay(self, states, actions_onehot, rewards, action_probs, dones, next_states):

        action_probs = np.vstack(action_probs)
        values = self.ac.critique(tf.convert_to_tensor(states, dtype=tf.float32))
        next_values = self.ac.critique(tf.convert_to_tensor(next_states, dtype=tf.float32))

        advantages, values_target = self.get_gaes(rewards, dones, values, next_values)

        loss_actor_avg = 0
        loss_critic_avg = 0
        for _ in range(self.replay_epochs):

            with tf.GradientTape() as tape:
                # action probs change as policy is updated
                new_action_probs = self.ac.act(tf.convert_to_tensor(states, dtype=tf.float32))
                #print(f"states={states[0:3]} action_probs={action_probs[0:3]} new_action_probs={new_action_probs[0:3]}", flush=True)
                actor_loss = self.ac.loss_actor(tf.convert_to_tensor(advantages, dtype=tf.float32),
                                                tf.convert_to_tensor(action_probs, dtype=tf.float32),
                                                tf.convert_to_tensor(actions_onehot, dtype=tf.float32),
                                                new_action_probs)

                values_pred = self.ac.critique(tf.convert_to_tensor(states, dtype=tf.float32))
                critic_loss = self.ac.loss_critic(values, values_target, values_pred)

                loss = actor_loss + critic_loss
                gradients = tape.gradient([loss], self.ac.trainable_variables)

            self.optimizer.apply_gradients(zip(gradients, self.ac.trainable_variables))

            loss_actor_avg += actor_loss
            loss_critic_avg += critic_loss

        loss_actor_avg /= self.replay_epochs
        loss_critic_avg /= self.replay_epochs
        return loss_actor_avg, loss_critic_avg


    def run(self):
        plot = Plot("Avg score")
        episode, score, done = 0, 0, False
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False
            states = np.empty((0, 1, len(state)))
            next_states = np.empty((0, 1, len(state)))
            actions = np.empty((0, self.action_size))
            action_probs = np.empty((0, self.action_size))
            rewards, dones = [], []
            num_steps = 0
            t_start = time.time()

            while not done:
                action, action_onehot, prob = self.action(state)
                #print(f"action={action} action_onehot={action_onehot} prob={prob}")
                next_state, reward, done, _ = self.env.step(action)
                num_steps += 1

                # reshape states to (batch=1, timesteps=1, num_features)
                states = np.append(states, np.reshape(state, (1,1,state.shape[0])), axis=0)
                next_states = np.append(next_states, np.reshape(next_state, (1, 1, state.shape[0])), axis=0)
                actions = np.append(actions, np.reshape(action_onehot, (1, self.action_size)), axis=0)
                action_probs = np.append(action_probs, prob, axis=0)
                rewards.append(reward)
                dones.append(done)
                state = next_state
                score += reward

                if done:
                    episode += 1
                    score_average = self.update_score(score)

                    if (score_average > self.max_score_avg):
                        self.max_score_avg = score_average * 1.2
                        self.lr = self.lr * 0.95;
                        K.set_value(self.optimizer.learning_rate, self.lr)
                        print(f"new max average={score_average}; new learning rate={self.lr:.5f}")
                        y1 = self.actor.model.predict(states[0])
                        self.actor.model.save("models/actor")
                        print("saved model as models/actor")

                        if self.play:
                            self.test()

                    if self.plot:
                        plot.update(score_average)

                    loss_actor, loss_critic = self.replay(states, actions, rewards, action_probs, dones, next_states)
                    t_end = time.time()
                    print("episode: {}/{} fps={:.2f} score: {:.2f}, average: {:.2f} loss actor={:.5f} critic={:.5f} num states={:d}"\
                          .format(episode,  self.num_episodes, num_steps / (t_end - t_start),score, score_average, loss_actor, loss_critic, len(states)))

                    state  = self.env.reset()
                    score = 0


    def test(self):
        state = self.env.reset()
        done = False
        while not done:
            action, _, _ = self.action(state)
            state, reward, done, _ = self.env.step(action)
            self.env.render()


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
    parser.add_argument('--no_recurrent', action='store_true', default=False, help='on: use Dense; off: use RNN for feature extraction')
    parser.add_argument('--no_play', action='store_true', default=False, help='play test')
    parser.add_argument('--no_plot', action='store_true', default=False, help='disable average score plot ')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--env', type=str, default="LunarLander-v2", help='gym environment name')
    args = parser.parse_args()
    args.plot = not args.no_plot
    args.play  = not args.no_play
    args.recurrent  = not args.no_recurrent
    return args


if __name__ == "__main__":

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

    agent = Agent(env_name=args.env, recurrent=args.recurrent, lr=args.lr, play=args.play, plot=args.plot)
    agent.run()
