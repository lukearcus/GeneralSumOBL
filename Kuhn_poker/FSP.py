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

def RL(RL_mem,Q, k):
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
    return beta, Q


class FSP:

    def __init__(self, _game, _RL_algo=RL, _SL_algo=count_based_SL, max_iters=100, m=50, n=50):
        self.game = _game
        self.RL = _RL_algo
        self.SL = _SL_algo
        self.num_players = self.game.num_players
        self.m = m
        self.n = n
        self.max_iters = max_iters
        self.exploitability_iters = 100
        self.est_exploit_freq = 10

    def gen_data(self, pi, beta, n, m, eta):
        sigma = []
        for p in range(self.num_players):
            sigma.append((1-eta)*pi[p]+eta*beta[p])
        D_mixed = []
        for i in range(n):
            D_mixed.append(self.play_game(sigma))
        D = []
        for p in range(self.num_players):
            D.append([])
            for i in range(m):
                strat = sigma.copy()
                strat[p] = beta[p]
                D[p].append((self.play_game(strat),strat[p]))
            D[p] += zip(D_mixed,[sigma[p] for j in D_mixed])
        return D

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
        exploitability = []
        for j in range(2,self.max_iters):
            eta_j = 1/j
            #eta_j = 1/2
            D = self.gen_data(pi[-1],beta[-1], self.n, self.m, eta_j)
            new_beta = []
            new_pi = []
            for p in range(self.num_players):
                mem_RL[p] = self.update_RL_mem(mem_RL[p], D[p], p)
                mem_SL[p] = self.update_SL_mem(mem_SL[p], D[p], p)
                new_b, Q[p] = self.RL(mem_RL[p],Q[p],j)
                new_beta.append(new_b)
                new_pi.append(self.SL(mem_SL[p]))
            pi.append(new_pi)
            beta.append(new_beta)
            if j%self.est_exploit_freq == 0:
                exploitability.append(self.est_exploitability(new_beta, new_pi))
        return pi[-1], exploitability, (pi, beta, Q, D)

    def update_RL_mem(self, old_mem, data, p):
        new_mem=old_mem
        for elem in data:
            new_mem += elem[0][p]
        
        #do we store our own behaviour (s_t, a_t, r_{t+1}, s_{t+1}) or other agents'????

        #for player in range(self.num_players):
        #    if player != p:
        #        for elem in data[player]:
        #            new_mem[player].append(elem)
        return new_mem

    def update_SL_mem(self, old_mem, data, p):
        new_mem=old_mem
        for elem in data:
            #Here we store our own behaviour tuple and the policy we were following?
            new_mem.append((elem[0][p], elem[1]))
        return new_mem

    def play_game(self, strat):
        buffer = [[] for i in range(self.num_players)]
        self.game.start_game()
        while not self.game.ended:
            curr_p = self.game.curr_player
            curr_p_strat = strat[curr_p]
            obs = self.game.observe()
            probs = curr_p_strat[obs,:]
            action = np.argmax(np.random.multinomial(1, pvals= probs))
            self.game.action(action)
            buffer[curr_p].append({'s':obs,'a':action,'r':0,"s'":-1})
            if len(buffer[curr_p]) > 1:
                buffer[curr_p][-2]["s'"] = obs
        rewards = self.game.end_game()
        for p in range(self.num_players):
            buffer[p][-1]["r"]=rewards[p]
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
