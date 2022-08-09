import FSP
import Kuhn_poker.game as game
import matplotlib.pyplot as plt
import agents.learners as learners
from UI.plot_funcs import FSP_plots 


#sort of working
extras = 20
num_BR = 30
num_mixed = 10
iters= 200000
time = 300

#test
#extras = 2
#num_BR = 4
#num_mixed = 0
#iters = 100000
#time = 60

KP_game = game.Kuhn_Poker_int_io()

RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage, 2, 6, init_adv=-2, extra_samples = extras)\
               for p in range(2)]
SL_learners = [learners.count_based_SL((6,2)) for p in range(2)]

agents = [learners.complete_learner(RL_learners[p], SL_learners[p]) for p in range(2)]

worker = FSP.FSP(KP_game, agents, max_iters=iters, max_time=time, m=num_BR, n=num_mixed)
pi, exploitability, data = worker.run_algo()

FSP_plots(exploitability, worker.est_exploit_freq, [pi], 'kuhn') 

import pdb; pdb.set_trace()
