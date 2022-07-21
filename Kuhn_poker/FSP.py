import numpy as np
import random

def count_based_SL(SL_mem):
    N = np.zeros(SL_mem[0][1].shape)
    for elem in SL_mem:
        for ep in elem[0]:
            state = ep['s']
            N[state,:] += elem[1][state,:]
    pi = N/np.sum(N,axis=1)[:,np.newaxis]
    return pi

def RL(RL_mem,Q):
    gamma = 0.9
    lr=0.1
    RL_buff = random.choices(RL_mem, k=min(100, len(RL_mem)))
    for elem in RL_buff:
        if elem["s'"] == -1:
            update = elem["r"] - Q[elem["s"],elem["a"]] 
        else:
            update = elem["r"] - Q[elem["s"],elem["a"]] + gamma*np.max(Q[elem["s"],:])
        Q[elem["s"],elem["a"]] += lr*update
    beta = np.zeros(Q.shape)
    for i, s in enumerate(Q):
        beta[i, np.argmax(s)] = 1
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


    def gen_data(self, pi, beta, n, m, nu):
        sigma = []
        for p in range(self.num_players):
            sigma.append((1-nu)*pi[p]+nu*beta[p])
        D_mixed = []
        for i in range(n):
            D_mixed.append(self.play_game(sigma))
        D = []
        for p in range(self.num_players):
            D.append([])
            for i in range(m):
                strat = sigma
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
        Q = np.copy(pi_1)
        for j in range(2,self.max_iters):
            nu_j = 1/j
            D = self.gen_data(pi[-1],beta[-1], self.n, self.m, nu_j)
            new_beta = []
            new_pi = []
            for p in range(self.num_players):
                mem_RL[p] = self.update_RL_mem(mem_RL[p], D[p], p)
                mem_SL[p] = self.update_SL_mem(mem_SL[p], D[p], p)
                new_b, Q[p] = self.RL(mem_RL[p],Q[p])
                new_beta.append(new_b)
                new_pi.append(self.SL(mem_SL[p]))
            pi.append(new_pi)
            beta.append(new_beta)
        return pi[-1]

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


