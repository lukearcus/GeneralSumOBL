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
    r_hist = []
    def observe(self, observation):
        if observation[0] != -1:
            print("Current pot is: " + str(observation[1]))
            print("Your card is: " + str(observation[0]))
        else:
            print("Your payout is: " + str(observation[2]))
        self.state = observation
        self.state_hist.append(self.state)
        self.r_hist.append(observation[2])

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

class Q_learn(player):

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
        self.next_state_hist = []
        self.r_hist = []
        self.tuple_hist = []

    def reset_Q(self):
        self.Q_mat = np.copy(self.init_Q_mat)

    def observe(self, observation):
        if observation[0] == -1:
            self.state = -1
        else:
            if observation[1][0] == observation[1][1]:
                self.state = observation[0] - 1
            else:
                self.state = 2 + observation[0]
        reward = observation[2]
        if len(self.state_hist) > 0:
            self.next_state_hist.append(self.state)
            s = self.state_hist[-1]
            a = self.act_hist[-1]
            r = reward
            s_prime = self.state
            self.tuple_hist.append((s, a, r, s_prime))
            self.single_Q_update(s, a, r, s_prime)
        self.state_hist.append(self.state)
        self.r_hist.append(reward)

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
    
    def single_Q_update(self, s, a, r, s_prime):
        if s_prime == -1:
            update = r - self.Q_mat[s,a]
        else:
            update = r - self.Q_mat[s,a] + self.d_f*np.max(self.Q_mat[s_prime])
        self.Q_mat[s, a] += self.lr * update

    def train_all(self):
        for i, elem in enumerate(self.tuple_hist):
            s, a, r, s_prime = elem
            self.single_Q_update(s, a, r, s_prime)

class OBL(Q_learn):
    belief = 0
    
    def single_Q_update(self, s, a, _1, _2):
        """
        Overwrite Q_update to do OBL
        """
        probs = self.belief[s]
        fict = np.argmax(np.random.multinomial(1, pvals = probs))

    def set_belief(self, new_belief):
        self.belief = new_belief

class actor_critic(Q_learn):
     
    def __init__(self, init_q, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.pi = np.ones((num_states, num_actions))*(1/num_actions)
        self.thetas = np.ones((num_states, 1))*(1/num_actions)
        self.critic = np.ones((num_states, num_actions))*init_q
        self.lr = learning_rate
        self.d_f = discount_factor
        self.eps = exploration_rate
        random.seed(1)
        self.reset()

    def action(self):
        if self.state < 3:
            acts = ["bet", "check"]
        else:
            acts = ["bet", "fold"]
        p_vals = np.copy(self.pi[self.state,:])
        if random.random() < self.eps:
            sel_act = np.random.randint(len(p_vals))
        else:
            sel_act = np.argmax(np.random.multinomial(1, pvals=p_vals))
        self.act_hist.append(sel_act)
        self.act = sel_act
        return acts[sel_act]

    def calc_grad(self, s, a):
        raise NotImplementedError

    def set_pi(self):
        raise NotImplementedError

    def calc_delta(self, s, a, r, s_prime):
        return NotImplementedError

    def single_Q_update(self, s, a, r, s_prime):
        """
        Overwrite Q_update to do actor-critic
        """
        #A_w = self.critic[s, a]
        A_w = self.calc_delta(s,a,r,s_prime)
        grad_log_theta = self.calc_grad(s, a)
        policy_update = A_w*grad_log_theta
        
        self.thetas = np.minimum(np.maximum(1e-5, self.thetas + self.lr * policy_update), 1-1e-5)
        
        self.set_pi()
        

        critic_update = self.calc_delta(s, a, r, s_prime)
        self.critic[s, a] += self.lr*critic_update

class actor_critic_lin_pol(actor_critic):

    def calc_grad(self, s, a):
        theta = self.pi[s, 0] # assume 2 actions with probs theta and 1-theta
        if a == 0:
            grad_log_theta = 1/theta
        else:
            grad_log_theta = -1/(1-theta)
        grad_log_theta_all = np.zeros(self.thetas.shape)
        grad_log_theta_all[s] = grad_log_theta
        return grad_log_theta_all
    
    def set_pi(self):
        self.pi[:,0] = self.thetas.flatten()
        self.pi[:,1] = 1-self.thetas.flatten()

class Q_actor_critic_lin_pol(actor_critic_lin_pol):

    def calc_delta(self, s, a, r, s_prime):
        if s_prime != -1:
            next_p_vals = self.pi[s_prime]
            a_prime = np.argmax(np.random.multinomial(1, pvals=next_p_vals))
            delta_t = r + self.d_f*self.critic[s_prime, a_prime] - self.critic[s, a]
        else:
            delta_t = r - self.critic[s,a]
        return delta_t

class advantage_actor_critic_lin_pol(actor_critic_lin_pol):
    
    def __init__(self, init_q, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        super().__init__(init_q, num_states, num_actions, learning_rate, discount_factor, exploration_rate)
        self.Value = np.zeros(num_states)

    def single_Q_update(self, s, a, r, s_prime):
        super().single_Q_update(s, a, r, s_prime)
        self.update_V(s, a, r, s_prime)

    def update_V(self, s, a, r, s_prime):
        if s_prime != -1:
            target = r + self.d_f*self.Value[s_prime]
        else:
            target = r
        error = target - self.Value[s]
        self.Value[s] += self.lr*error

    def calc_delta(self, s, a, r, s_prime):
        if s_prime != -1:
            delta_t = r + self.d_f*self.Value[s_prime] - self.Value[s]
        else:
            delta_t = r - self.Value[s]
        return delta_t
