import numpy as np
import random

class player:
    state = None
    belief = None

    def __init__(self):
        pass

    def observe(self, observation):
        pass

    def action(self):
        sel_act = None
        return sel_act
    
    def reset(self):
        pass

class human(player):
    Q_mat = 0
    state_hist=[]

    def observe(self, observation):
        print("Current pot is: " + str(observation[1]))
        print("Your card is: " + str(observation[0]))
        self.state = observation
        self.state_hist.append(self.state)

    def action(self):
        print("Choose an action: ")
        if self.state[1][0] == self.state[1][1]:
            print("bet")
            print("check")
        else:
            print("bet")
            print("fold")
        sel_act = input()
        return sel_act
    
    def get_reward(self, reward):
        print("You got " + str(reward) + " coins")

class vanilla_rl(player):

    def __init__(self, init_q, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3, T=0.01):
        self.Q_mat = np.ones((num_states, num_actions))*init_q
        self.init_Q_mat = np.copy(self.Q_mat)
        self.lr = learning_rate
        self.d_f = discount_factor
        self.eps = exploration_rate
        self.temp = T
        random.seed(1)
        self.reset()

    def reset(self):
        self.state_hist = []
        self.act_hist = []

    def reset_Q(self):
        self.Q_mat = np.copy(self.init_Q_mat)

    def observe(self, observation):
        if observation[1][0] == observation[1][1]:
            self.state = observation[0] - 1
        else:
            self.state = 2 + observation[0]
        self.state_hist.append(self.state)

    def action(self):
        if self.state < 3:
            acts = ["bet", "check"]
        else:
            acts = ["bet", "fold"]
        q_vals = np.copy(self.Q_mat[self.state,:])
        if random.random() < self.eps:
            sel_act = np.random.randint(len(q_vals))
        else:
            q_vals -= np.max(q_vals)
            q_vals = np.exp(self.temp*q_vals)
            q_vals /= np.sum(q_vals)
            sel_act = np.argmax(np.random.multinomial(1, pvals=q_vals))
        self.act_hist.append(sel_act)
        self.act = sel_act
        return acts[sel_act]
    
    def Q_update(self, s, a, r, s_prime):
        if s_prime == -1:
            update = r - self.Q_mat[s,a]
        else:
            update = r - self.Q_mat[s,a] + self.d_f*np.max(self.Q_mat[s_prime])
        self.Q_mat[s, a] += self.lr * update

    def get_reward(self, reward):
        self.state_hist.reverse()
        self.act_hist.reverse()
        r_hist = [0 for i in self.act_hist]
        r_hist[0] = reward
        next_states = [s for s in self.state_hist]
        next_states.pop(0)
        next_states.append(-1)
        hist = zip(self.state_hist, self.act_hist, r_hist, next_states)
        for i, elem in enumerate(hist):
            s, a, r, s_prime = elem
            self.Q_update(s, a, r, s_prime)

class OBL(vanilla_rl):
    belief = 0

    def set_belief(self, new_belief):
        self.belief = new_belief
