import numpy as np
import random

def count_based_SL(SL_mem):
    N = np.zeros(SL_mem[0][1].shape)
    for elem in SL_mem:
        for ep in elem[0]:
            state = ep['s']
            N[state,:] += elem[1][state,:]
    pi = N/np.sum(N,axis=1)[:,np.newaxis]
    for i, s in enumerate(pi):
        if np.all(np.isnan(s)):
            pi[i] = np.ones(s.shape)/s.size
    return pi

def RL(RL_mem,Q, V, k, thetas):
    gamma = 0.9
    lr=0.05/(1+0.003*np.sqrt(k))
    T = 1/(1+0.2*np.sqrt(k))
    RL_buff = random.choices(RL_mem, k=min(30, len(RL_mem)))
    for elem in RL_buff:
        if elem["s'"] == -1:
            update = elem["r"] - Q[elem["s"],elem["a"]] 
        else:
            update = elem["r"] - Q[elem["s"],elem["a"]] + gamma*np.max(Q[elem["s'"],:])
        Q[elem["s"],elem["a"]] += lr*update
    beta = np.zeros(Q.shape)
    for i, s in enumerate(Q):
        beta[i, :] = np.exp(Q[i]/T)
        beta[i, :] /= np.sum(beta[i])
    return beta, Q, V, thetas



def adv_actor_critic_softmax(RL_mem, Q, V, k, thetas):
    gamma=0.9
    lr=0.05/(1+0.003*np.sqrt(k))

    RL_buff = random.choices(RL_mem, k=min(300, len(RL_mem)))
    for elem in RL_buff:
        theta = thetas[elem["s"], elem["a"]]
        grad_log_theta = 1-np.exp(theta)/np.sum(np.exp(thetas[elem["s"],:]))

        if elem["s'"] == -1:
            advantage = elem["r"] - V[elem["s"]]
        else:
            advantage = elem["r"] - V[elem["s"]] + gamma*V[elem["s'"]]
        theta_update = grad_log_theta*advantage
        thetas[elem["s"],elem["a"]] = theta + lr*theta_update
    
    for elem in RL_buff:
        if elem["s'"] == -1:
            update = elem["r"] - V[elem["s"]]
        else:
            update = elem["r"] - V[elem["s"]] + gamma*V[elem["s'"]]
        V[elem["s"]] += lr*update
    beta = np.exp(thetas)/np.sum(np.exp(thetas),axis=1)[:,np.newaxis]
    
    return beta, Q, V, thetas

def adv_actor_critic(RL_mem, Q, V, k, thetas):
    gamma=0.9
    lr=0.05/(1+0.003*np.sqrt(k))

    RL_buff = random.choices(RL_mem, k=min(300, len(RL_mem)))
    for elem in RL_buff:
        theta = thetas[elem["s"]]
        if elem["a"] == 0:
            grad_log_theta = 1/theta
        else:
            grad_log_theta = -1/(1-theta)
        if elem["s'"] == -1:
            advantage = elem["r"] - V[elem["s"]]
        else:
            advantage = elem["r"] - V[elem["s"]] + gamma*V[elem["s'"]]
        theta_update = grad_log_theta*advantage
        thetas[elem["s"]] = np.minimum(np.maximum(theta + lr*theta_update, 1e-5),1-1e-5)
    
    for elem in RL_buff:
        if elem["s'"] == -1:
            update = elem["r"] - V[elem["s"]]
        else:
            update = elem["r"] - V[elem["s"]] + gamma*V[elem["s'"]]
        V[elem["s"]] += lr*update
    beta = np.array([thetas, 1-thetas]).T
    
    return beta, Q, V, thetas


class FSP:

    def __init__(self, _game, _RL_algo=adv_actor_critic_softmax, _SL_algo=count_based_SL, max_iters=100, m=50, n=50):
        self.game = _game
        self.RL = _RL_algo
        self.SL = _SL_algo
        self.num_players = self.game.num_players
        self.m = m
        self.n = n
        self.max_iters = max_iters
        self.exploitability_iters = 1000
        self.est_exploit_freq = 100

    def gen_data(self, pi, beta, n, m, eta):
        sigma = []
        for p in range(self.num_players):
            sigma.append((1-eta)*pi[p]+eta*beta[p])
        D_mixed = []
        for i in range(n):
            D_mixed.append(self.play_game(sigma))
        D = []
        exploitability = 0
        for p in range(self.num_players):
            D.append([])
            for i in range(m):
                strat = sigma.copy()
                strat[p] = beta[p]
                result = self.play_game(strat)
                exploitability += result[p][-1]['r']/(m*self.num_players)
                D[p].append((result,strat[p]))
            D[p] += zip(D_mixed,[sigma[p] for j in D_mixed])
        return D, exploitability

    def run_algo(self):
        pi = []
        beta = []
        pi_1 = []
        for p in range(self.num_players):
           player_pol = np.ones([self.game.num_states[p], self.game.num_actions[p]]) * (1/self.game.num_actions[p])
           pi_1.append(player_pol)
        pi.append(pi_1)
        beta.append(pi_1)

        mem_RL = [[] for i in range(self.num_players)]
        mem_SL = [[] for i in range(self.num_players)]
        Q = [np.zeros(pi_1[i].shape) for i in range(self.num_players)]
        V = [np.zeros(self.game.num_states[p]) for p in range(self.num_players)]
        #thetas = [np.ones(self.game.num_states[p])/self.game.num_actions[p] for p in range(self.num_players)]
        thetas = [np.ones((self.game.num_states[p], self.game.num_actions[p])) for p in range(self.num_players)]
        exploitability = []
        for j in range(2,self.max_iters):
            eta_j = 1/j
            #eta_j = 1/2
            D, curr_exploitability = self.gen_data(pi[-1],beta[-1], self.n, self.m, eta_j)
            exploitability.append(curr_exploitability)
            new_beta = []
            new_pi = []
            for p in range(self.num_players):
                mem_RL[p] = self.update_RL_mem(mem_RL[p], D[p], p)
                mem_SL[p] = self.update_SL_mem(mem_SL[p], D[p], p, beta[-1][p])
                new_b, Q[p], V[p], thetas[p] = self.RL(mem_RL[p],Q[p],V[p],j, thetas[p])
                new_beta.append(new_b)
                new_pi.append(self.SL(mem_SL[p]))
            pi.append(new_pi)
            beta.append(new_beta)
            
            #if j%self.est_exploit_freq == 0:
            #    exploitability.append(self.est_exploitability(new_beta, new_pi))
        import pdb; pdb.set_trace()
        return pi[-1], exploitability, (pi, beta, Q, D)

    def update_RL_mem(self, old_mem, data, p):
        new_mem=old_mem
        for elem in data:
            new_mem += elem[0][p]
        return new_mem

    def update_SL_mem(self, old_mem, data, p, beta):
        new_mem=old_mem
        for elem in data:
            #Here we store our own behaviour tuple and the policy we were following?
            if np.all(elem[1] == beta):
                new_mem.append((elem[0][p], elem[1]))
        return new_mem

    def play_game(self, strat):
        buffer = [[] for i in range(self.num_players)]
        self.game.start_game()
        while not self.game.ended:
            curr_p = self.game.curr_player
            curr_p_strat = strat[curr_p]
            obs, r = self.game.observe()
            probs = curr_p_strat[obs,:]
            action = np.argmax(np.random.multinomial(1, pvals= probs))
            self.game.action(action)
            buffer[curr_p].append({'s':obs,'a':action,'r':r,"s'":-1})
            if len(buffer[curr_p]) > 1:
                buffer[curr_p][-2]["s'"] = obs
         
        for i in range(self.num_players):
            player = self.game.curr_player
            _, r = self.game.observe()
            self.game.action(None)
            buffer[player][-1]["r"] = r
        return buffer

    def est_exploitability(self, beta, pi):
        R = [0 for i in range(self.num_players)]
        for i in range(self.exploitability_iters):
            for p in range(self.num_players):
                strat = pi.copy()
                strat[p] = beta[p]
                buff = self.play_game(strat)
                R[p] += buff[p][-1]["r"]
        
        for p in range(self.num_players):
            R[p] /= self.exploitability_iters

        return sum(R)
