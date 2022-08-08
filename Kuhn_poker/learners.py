import random
import numpy as np

class learner_base:
    iteration = 0
    num_samples = 100

    def update_memory(self, data):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def wipe_memory(self):
        self.memory = []

class RL_base(learner_base):
    opt_pol = None
    memory = []
    
    def __init__(self, extra_samples, init_lr, df):
        self.extra_samples = extra_samples
        self.gamma = df
        self.init_lr = init_lr

    def update_memory(self, data):
        self.last_round = len(data)
        for elem in data:
            self.memory += elem[0]

class SL_base(learner_base):
    learned_pol = None
    memory = []

    def __init__(self, pol_shape):
        self.learned_pol = np.ones(pol_shape)/pol_shape[1]

    def update_memory(self, data):
        for elem in data:
            if elem[2]:
                self.memory.append((elem[0], elem[1]))

class count_based_SL(SL_base):

    def __init__(self, pol_shape):
        self.memory = []
        super().__init__(pol_shape)

    def learn(self):
        N = np.zeros(self.memory[0][1].shape)
        for elem in self.memory:
            for ep in elem[0]:
                state = ep['s']
                N[state,:] += elem[1][state,:]
        pi = N/np.sum(N,axis=1)[:,np.newaxis]
        for i, s in enumerate(pi):
            if np.all(np.isnan(s)):
                pi[i] = np.ones(s.shape)/s.size
        self.learned_pol = pi
        return pi

class Q_learn(RL_base):

    def __init__(self, init_q, Q_shape, num_samples=100, init_lr = 0.05, df=1.0):
        self.Q = np.ones(Q_shape)*init_q
        self.calc_pol()
        super().__init__(num_samples, init_lr, df)
    
    def calc_pol(self):
        beta = np.zeros(self.Q.shape)
        for i, s in enumerate(self.Q):
            beta[i, :] = np.exp(self.Q[i]/T)
            beta[i, :] /= np.sum(beta[i])
        self.opt_pol = beta

    def learn(self):
        self.iteration += 1
        lr = self.init_lr/(1+0.003*np.sqrt(self.iteration))
        T = 1/(1+0.2*np.sqrt(self.iteration))

        RL_buff = random.sample(self.memory, min(self.num_samples, len(self.memory)))
        for elem in RL_buff:
            if elem["s'"] == -1:
                update = elem["r"] - self.Q[elem["s"],elem["a"]] 
            else:
                update = elem["r"] - self.Q[elem["s"],elem["a"]] \
                        + self.gamma*np.max(self.Q[elem["s'"],:])
            self.Q[elem["s"],elem["a"]] += lr*update
        self.calc_pol()
        return self.opt_pol


class actor_critic(RL_base):

    def __init__(self, pol_func, advantage_func, num_actions, num_states, init_adv = 0, extra_samples=10, init_lr=0.05, df=1.0):
        self.pol_func = pol_func(num_states, num_actions)
        self.advantage_func = advantage_func(init_adv, num_states, num_actions, df)
        self.opt_pol = self.pol_func.policy
        self.memory = []
        super().__init__(extra_samples, init_lr, df)

    def reset(self):
        self.pol_func.reset()
        self.advantage_func.reset()

    def learn(self):
        self.iteration += 1
        lr = self.init_lr/(1+0.003*np.sqrt(self.iteration))
        RL_buff = random.sample(self.memory, min(self.extra_samples, len(self.memory)))
        RL_buff += self.memory[-min(self.last_round, len(self.memory)):]
        
        for elem in RL_buff:
            theta_update = np.zeros_like(self.pol_func.thetas)
            grad_log_theta = self.pol_func.grad_log(elem["s"], elem["a"])
            advantage = self.advantage_func.eval(elem["s"], elem["a"], elem["r"], elem["s'"])
            theta_update += grad_log_theta*advantage
            
            a_prime = np.argmax(np.random.multinomial(1, pvals=self.opt_pol[elem["s'"]]))
            delta = self.advantage_func.calc_delta(elem["s"], elem["a"], elem["r"], elem["s'"], a_prime)
            self.advantage_func.update(lr*delta, elem["s"], elem["a"])
            self.pol_func.thetas += lr*theta_update
        self.opt_pol = self.pol_func.update()
        return self.opt_pol

class pol_func_base:
    thetas = None
    policy = None

    def __init__(self, num_states, num_actions):
        raise NotImplementedError

    def grad_log(self, s, a):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

class advantage_func_base:

    def __init__(self, init_adv, num_states, num_actions):
        raise NotImplementedError

    def eval(self, s, a, r, s_prime):
        raise NotImplementedError
    
    def calc_delta(self, s, a, r, s_prime, a_prime):
        return self.eval(s, a, r, s_prime)

    def update(self, update, s, a):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

class softmax(pol_func_base):

    def __init__(self, num_states, num_actions):
        self.thetas = np.ones((num_states, num_actions))
        self.update()

    def grad_log(self, s, a):
        grad = np.zeros_like(self.thetas)
        for act in range(self.thetas.shape[1]):
            if act == a:
                grad[s,act] = 1-self.policy[s,act]
            else:
                grad[s,act] = -self.policy[s,act]
        return grad

    def update(self):
        self.policy = np.exp(self.thetas)/np.sum(np.exp(self.thetas),axis=1)[:,np.newaxis]
        return self.policy
    
    def reset(self):
        self.thetas = np.ones_like(self.thetas)
        self.update()

class Q_advantage(advantage_func_base):
    
    def __init__(self, init_adv, num_states, num_actions, df):
        self.Q = np.ones((num_states, num_actions))*init_adv
        self.V = np.ones(num_states)*init_adv
        self.init_adv = init_adv
        self.gamma = df

    def eval(self, s, a, r, s_prime):
        return self.Q[s,a]-self.V[s]
    
    def calc_delta(self, s, a, r, s_prime, a_prime):
        delta = np.zeros(2)
        if s_prime != -1:
            delta[0] = r + self.gamma*np.max(self.Q[s_prime, :]) - self.Q[s, a]
            delta[1] = r + self.gamma*self.V[s_prime] - self.V[s]
        else:
            delta[0] = r - self.Q[s,a]
            delta[1] = r - self.V[s]
        return delta

    def update(self, update, s, a):
        self.Q[s, a] += update[0]
        self.V[s] += update[1]

    def reset(self):
        self.Q = np.ones_like(self.Q) * self.init_adv
        self.V = np.ones_like(self.V) * self.init_adv

class Q_actor_critic(advantage_func_base):

    def __init__(self, init_adv, num_states, num_actions, df):
        self.Q = np.ones((num_states, num_actions))*init_adv
        self.init_adv = init_adv
        self.gamma = df

    def eval(self, s, a, r, s_prime):
        return self.Q[s,a]
    
    def calc_delta(self, s, a, r, s_prime, a_prime):
        if s_prime != -1:
            delta = r + self.gamma*self.Q[s_prime, a_prime] - self.Q[s, a]
        else:
            delta = r - self.Q[s,a]
        return delta

    def update(self, update, s, a):
        self.Q[s, a] += update

    def reset(self):
        self.Q = np.ones_like(self.Q) * self.init_adv


class value_advantage(advantage_func_base):

    def __init__(self, init_adv, num_states, _, df):
        self.V = np.ones(num_states)*init_adv
        self.init_adv = init_adv
        self.gamma = df

    def eval(self, s, a, r, s_prime):
        if s_prime == -1:
            advantage = r - self.V[s]
        else:
            advantage = r - self.V[s] + self.gamma*self.V[s_prime]
        return advantage

    def update(self, update, s, _):
        self.V[s] += update

    def reset(self):
        self.V = np.ones_like(self.V) * self.init_adv

class complete_learner:

    def __init__(self, RL, SL):
        self.RL_learner = RL
        self.SL_learner = SL
        self.beta = RL.opt_pol
        self.pi = SL.learned_pol

    def update_memory(self, data):
        self.RL_learner.update_memory(data)
        self.SL_learner.update_memory(data)

    def learn(self):
        self.RL_learner.reset()
        for i in range(100):
            self.beta = self.RL_learner.learn()
        self.pi = self.SL_learner.learn()
        return self.beta, self.pi
