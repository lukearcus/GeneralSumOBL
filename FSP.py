import numpy as np
import random
import time
import logging
log = logging.getLogger(__name__)

class FSP:

    def __init__(self, _game, _agents, max_iters=100, max_time=300, m=50, n=50, exploit_iters=100, exploit_freq=10):
        self.game = _game
        self.agents = _agents
        self.num_players = self.game.num_players
        self.m = m
        self.n = n
        self.max_iters = max_iters
        self.max_time = max_time
        self.exploitability_iters = exploit_iters
        self.est_exploit_freq = exploit_freq

    def gen_data(self, pi, beta, eta):
        sigma = []
        for p in range(self.num_players):
            sigma.append((1-eta)*pi[p]+eta*beta[p])
        D = [[] for i in range(self.num_players)]
        for i in range(self.n):
            res = self.play_game(sigma)
            for p in range(self.num_players):
                D[p].append((res[p],sigma[p],False))
        exploitability = 0
        for p in range(self.num_players):
            for i in range(self.m):
                strat = sigma.copy()
                strat[p] = beta[p]
                result = self.play_game(strat)
                exploitability += result[p][-1]['r']/(self.m)
                D[p].append((result[p],strat[p],True))
        return D, exploitability, sigma

    def run_algo(self):
        pi = []
        beta = []
        pi_1 = []
        for p in range(self.num_players):
           pi_1.append(self.agents[p].pi)

        pi.append(pi_1) # pi_1
        beta.append(pi_1) # beta_2

        exploitability = []
        tic = time.perf_counter()
        for j in range(2,self.max_iters):
            eta_j = 1/j
            #eta_j = 1/2
            D, curr_exploitability, sigma = self.gen_data(pi[-1],beta[-1], eta_j)
            #exploitability.append(curr_exploitability)
            new_beta = []
            new_pi = []
            diff = 0
            for p in range(self.num_players):
                self.agents[p].update_memory(D[p])
                new_b, new_p = self.agents[p].learn()
                new_beta.append(new_b) # beta_(j+1)
                new_pi.append(new_p) # pi_j
                log.debug("p" + str(p+1) + " sigma: " + str(sigma[p]))
                log.debug("p" + str(p+1) + " new_pi: " + str(new_pi[p]))
                log.debug("p" + str(p+1) + " new_beta: " + str(new_beta[p]))
                #import pdb; pdb.set_trace()
                diff += np.linalg.norm(new_pi[p]-sigma[p])
            log.info("norm difference between new_pi and sigma: " +str(diff))
            pi.append(new_pi)
            beta.append(new_beta)
            #import pdb; pdb.set_trace()
            if j%self.est_exploit_freq == 0:
                exploitability.append(self.est_exploitability(sigma, new_beta))
            toc = time.perf_counter()
            if toc-tic > self.max_time:
                break
        #import pdb; pdb.set_trace()
        return pi[-1], exploitability, (pi, beta, D)

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

    #def calc_true_BRs(self, pol):

        #for each information state
            #calc next state probs (given fixed opponent)
    #    if self.num_players != 2:
    #        raise NotImplementedError
    #    else:
    #            
    #    for player in range(self.num_players):
            

    def est_exploitability(self, pi, beta):
        #BRs = self.calc_BRs(pi)
        R = [0 for i in range(self.num_players)]
        for i in range(self.exploitability_iters):
            for p in range(self.num_players):
                strat = pi.copy()
                strat[p] = beta[p]
                buff = self.play_game(strat)
                R[p] += buff[p][-1]["r"]
        
        for p in range(self.num_players):
            R[p] /= self.exploitability_iters
        #import pdb; pdb.set_trace()
        log.info("Exploitability: " + str(sum(R)))
        return sum(R)
