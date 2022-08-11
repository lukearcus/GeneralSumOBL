import FSP
import Kuhn_poker.game as game
import matplotlib.pyplot as plt
import agents.learners as learners
from UI.plot_funcs import FSP_plots 
import logging

#sort of working
logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
extras = 20
num_BR = 30
num_mixed = 10
iters= 200000
time = 30

#test
extras = 0
num_BR = 5000
num_mixed = 5000
iters = 100000
time = 300

KP_game = game.Kuhn_Poker_int_io()

RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage, 2, 6, init_adv=0, extra_samples = extras)\
               for p in range(2)]
SL_learners = [learners.count_based_SL((6,2)) for p in range(2)]

agents = [learners.complete_learner(RL_learners[p], SL_learners[p]) for p in range(2)]

worker = FSP.FSP(KP_game, agents, max_iters=iters, max_time=time, m=num_BR, n=num_mixed, exploit_freq=1)
pi, exploitability, data = worker.run_algo()

FSP_plots(exploitability, worker.est_exploit_freq, [pi], 'kuhn') 

import pdb; pdb.set_trace()
