import numpy as np
import pandas as pd
import nashpy


class QLearningTable:
    def __init__(self, actions, joint, start, learning_rate=0.85, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions      # actions of this agent
        self.joint_actions = joint
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.joint_actions, dtype=np.float64)   # Q Table should use joint actions

    def update_epsilon(self):
        '''Execute epsilon decay. The minimum epsilon is set to 0.1.'''
        if self.epsilon > 0.1:
            self.epsilon *= 0.999
        else:
            self.epsilon = 0.1

    def choose_action(self, observation, pi):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() > self.epsilon:
            # choose best action
            state_action = pd.Series(pi)
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action


    def learn(self, s, a, r, s_, pis, done=False):
        '''
        update Q table using formula Q(s, a) += alpha * (R + gamma 
        * max_a' Q(s', a') - Q(s, a))
        '''
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s][a]
        if not done:
            # Calculate nashq
            nashQ = self.compute_nashq(s_, pis[0], pis[1])
            q_target = r + self.gamma * nashQ
        else:
            q_target = r 
        self.q_table.loc[s][a] += self.lr * (q_target - q_predict) 

    def compute_nashq(self, state, pi1, pi2):
        """
            compute nash q value
        """
        nashq = 0
        for (action1, action2) in self.joint_actions:
            nashq += pi1[action1] * pi2[action2] * self.q_table.loc[state][(action1, action2)]
        return nashq

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.joint_actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


def Nash(state, Agent1, Agent2):
    '''Calculate nash equilibrium on current state and Q(s) as rewards.'''
    Agent1.check_state_exist(state)
    Agent2.check_state_exist(state)
    q_1, q_2 = [], []
    for action1 in Agent1.actions:
        row_q_1, row_q_2 = [], []
        for action2 in Agent2.actions:
            joint_action = (action1, action2)
            row_q_1.append(Agent1.q_table.loc[state][joint_action])
            row_q_2.append(Agent2.q_table.loc[state][joint_action])
        q_1.append(row_q_1)
        q_2.append(row_q_2)
    
    game = nashpy.Game(np.array(q_1), np.array(q_2))
    equilibria = game.lemke_howson_enumeration()
    eq_list = list(equilibria)

    pi = None
    for eq in eq_list:
        if eq[0].shape == (len(Agent1.actions), ) and eq[1].shape == (len(Agent2.actions), ):
            if any(np.isnan(eq[0])) is False and any(np.isnan(eq[1])) is False:
                pi = eq
                break       
    if pi is None:          
        pi1 = np.repeat(
            1.0 / len(Agent1.actions), len(Agent1.actions))
        pi2 = np.repeat(
            1.0 / len(Agent2.actions), len(Agent2.actions))
        pi = (pi1, pi2)

    return pi[0], pi[1]

