import numpy as np
import random
import time
import logging
from agents import learners
from functions import calc_exploitability
from games.kuhn import Kuhn_Poker_int_io as kuhn
log = logging.getLogger(__name__)

class FSP:

    def __init__(self, _game, _agents, max_iters=100, max_time=300, m=50, n=50, exploit_iters=100000, exploit_freq=10):
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
        """
        Generate data using average policy pi and best response beta.
        Plays n games where both players use their average policy,
        and m games where each takes it in turn playing their current br.
        """
        #import pdb; pdb.set_trace()
        D = [[] for i in range(self.num_players)]
        for i in range(self.n):
            strat = []
            for p in range(self.num_players):
                if random.random() > eta:
                    strat.append(pi[p])
                else:
                    strat.append(beta[p])
            res = self.play_game(strat)
            for p in range(self.num_players):
                D[p].append((res[p],strat[p],False))
        exploitability = 0
        for p in range(self.num_players):
            for i in range(self.m):
                strat = []
                for p_id in range(self.num_players):
                    if random.random() > eta:
                        strat.append(pi[p_id])
                    else:
                        strat.append(beta[p_id])
                strat[p] = beta[p]
                result = self.play_game(strat)
                #exploitability += result[p][-1]['r']/(self.m)
                D[p].append((result[p],strat[p],True))
        return D

    def run_algo(self):
        """
        Main function for implementing FSP,
        returns final average policies, exploitability across iterations and some extra data.
        """
        pi = []
        beta = []
        pi_1 = []
        for p in range(self.num_players):
           pi_1.append(self.agents[p].pi)

        pi.append(pi_1) # pi_1
        beta.append(pi_1) # beta_2

        exploitability = []
        tic = time.perf_counter()
        if isinstance(self.game, kuhn):
            exploit_learner = learners.kuhn_exact_solver()
        else:
            #exploit_learner = learners.fitted_Q_iteration(0, (self.game.num_states[0], self.game.num_actions[0])) 
            exploit_learner = learners.actor_critic(learners.softmax, learners.value_advantage, \
                                                self.game.num_actions[0], self.game.num_states[0]) 
        times = []
        for j in range(1,self.max_iters): # start from 1 or 2?
            log.info("Iteration " + str(j))
            eta_j = 1/j
            D = self.gen_data(pi[-1],beta[-1], eta_j)
            new_beta = []
            new_pi = []
            diff = 0
            for p in range(self.num_players):
                self.agents[p].update_memory(D[p])
                new_b, new_p = self.agents[p].learn()
                new_beta.append(new_b) # beta_(j+1)
                new_pi.append(new_p) # pi_j
                log.debug("p" + str(p+1) + " new_pi: " + str(new_pi[p]))
                log.debug("p" + str(p+1) + " new_beta: " + str(new_beta[p]))
            pi.append(new_pi)
            beta.append(new_beta)
            if j%self.est_exploit_freq == 0:
                results = {'true' : [], 'est':[], 'beta': []}
                
                exploit, br_pols, _, values = calc_exploitability(new_pi, self.game, exploit_learner,\
                                                                            num_iters = 10**4, num_exploit_iters=10**4)
                log.info("exploitability: " + str(exploit))
                exploitability.append(exploit)
            toc = time.perf_counter()
            times.append(toc-tic)
            if toc-tic > self.max_time:
                break
        return pi[-1], exploitability, {'pi': pi, 'beta':beta, 'D': D, 'times':times}

    def play_game(self, strat):
        """
        Play through the chosen game using the strategies provided in strat,
        returns an experience buffer for both players, consisting of s,a,r,s' tuples
        """
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
            #import pdb; pdb.set_trace()
            if len(buffer[player]) > 0:
                buffer[player][-1]["r"] = r

        return buffer
