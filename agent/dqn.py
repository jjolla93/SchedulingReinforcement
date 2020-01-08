"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import tensorflow as tf
import os

import environment.scheduling as sch
import environment.work as work
from agent.helper import *

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            feature_size,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=True,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.feature_size = feature_size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/dqn/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.s_reshaped = tf.reshape(self.s, shape=[-1, self.feature_size[0], self.feature_size[1], 1])
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first convolution layer.
            with tf.variable_scope('conv1'):
                w1_conv = tf.get_variable('w1_conv', [4, 4, 1, 16], initializer=w_initializer, collections=c_names)
                b1_conv = tf.get_variable('b1_conv', [1], initializer=b_initializer, collections=c_names)
                conv1 = tf.nn.relu(
                    tf.nn.conv2d(self.s_reshaped, w1_conv, strides=[1, 2, 2, 1], padding='SAME') + b1_conv)

            # second convolution layer.
            with tf.variable_scope('conv2'):
                w2_conv = tf.get_variable('w2_conv', [2, 2, 16, 32], initializer=w_initializer, collections=c_names)
                b2_conv = tf.get_variable('b2_conv', [1], initializer=b_initializer, collections=c_names)
                conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2_conv, strides=[1, 1, 1, 1], padding='SAME') + b2_conv)
                conv2 = tf.layers.flatten(conv2)

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [11872, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(conv2, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        self.s_reshaped_ = tf.reshape(self.s, shape=[-1, self.feature_size[0], self.feature_size[1], 1])
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first convolution layer.
            with tf.variable_scope('conv1'):
                w1_conv = tf.get_variable('w1_conv', [4, 4, 1, 16], initializer=w_initializer, collections=c_names)
                b1_conv = tf.get_variable('b1_conv', [1], initializer=b_initializer, collections=c_names)
                conv1 = tf.nn.relu(
                    tf.nn.conv2d(self.s_reshaped_, w1_conv, strides=[1, 2, 2, 1], padding='SAME') + b1_conv)

            # second convolution layer.
            with tf.variable_scope('conv2'):
                w2_conv = tf.get_variable('w2_conv', [2, 2, 16, 32], initializer=w_initializer, collections=c_names)
                b2_conv = tf.get_variable('b2_conv', [1], initializer=b_initializer, collections=c_names)
                conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2_conv, strides=[1, 1, 1, 1], padding='SAME') + b2_conv)
                conv2 = tf.layers.flatten(conv2)

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [11872, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(conv2, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            #print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def plot_reward(rewards):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(rewards)), rewards)
    plt.ylabel('average reward')
    plt.xlabel('training episode')
    plt.show()


def run(episodes=1000, update_term=5):
    step = 0
    rewards = []
    avg_rewards = []
    for episode in range(episodes):
        # initial observation
        observation = env.reset()
        rs = []
        actions = []
        episode_frames = []
        while True:
            episode_frames.append(observation)
            # RL choose action based on observation
            action = RL.choose_action(observation)
            actions.append(action)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            rs.append(reward)
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % update_term == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                rewards.append(sum(rs))
                avg_rewards.append(sum(rewards) / len(rewards))
                #if len(rewards) > 100:
                    #avg_rewards.append(sum(rewards[-100:]) / 100)
                if episode > 8000 and episode % 100 == 0 and False:
                    print(actions)
                    print(rs)
                    print(sum(rs))
                break
            step += 1
        if episode % 100 == 0 and episode != 0:
            print('episode: {0} finished'.format(episode))
            save_gif(episode_frames, RL.feature_size, episode, 'dqn')

    plot_reward(avg_rewards)
    # end of game
    print('game over')


if __name__ == "__main__":
    #days = 15  # 90일 231
    #blocks = 5  # 10개 444
    '''
    inbounds = [work.Work('Work0', 0, 2, -1, -1, days), work.Work('Work1', 0, 1, -1, 5, days),
                work.Work('Work2', 1, 1, 1, -1, days), work.Work('Work3', 1, 3, -1, 8, days),
                work.Work('Work4', 2, 2, 4, -1, days), work.Work('Work5', 2, 2, -1, 10, days),
                work.Work('Work6', 3, 1, 2, -1, days), work.Work('Work7', 3, 3, -1, 12, days),
                work.Work('Work8', 4, 2, 4, -1, days), work.Work('Work9', 4, 2, -1, 14, days)]
    # inbounds.append(work.Work('Work9', 4, 2, -1, 14, days))
    '''

    inbounds = work.import_blocks_schedule('../environment/data/191227_납기일 추가.xlsx')

    '''
    for i in range(len(inbounds)):
        print("블록 번호: {0}, 계획공기: {1}, 납기일: {2}".format(inbounds[i].block, inbounds[i].lead_time, inbounds[i].latest_finish))
    '''

    blocks = inbounds[-1].block + 1
    days = inbounds[-1].latest_finish + 1

    if not os.path.exists('../frames/dqn'):
        os.makedirs('../frames/dqn')
    if not os.path.exists('../frames/dqn/%d-%d' % (days, blocks)):
        os.makedirs('../frames/dqn/%d-%d' % (days, blocks))

    env = sch.Scheduling(num_days=days, num_blocks=blocks, inbound_works=inbounds, display_env=False)
    RL = DeepQNetwork(env.action_space, env.n_features, (env.num_block, env.num_days),
                      learning_rate=1e-4,
                      reward_decay=0.99,
                      e_greedy=0.7,
                      replace_target_iter=200,
                      memory_size=50000,
                      output_graph=False
                      #,e_greedy_increment=-1e-5
                      )
    run(5000)
