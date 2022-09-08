import FSP
import games.leduc as leduc
import games.kuhn as KP
import matplotlib.pyplot as plt
import agents.learners as learners
import UI.get_args as get_args
from UI.plot_funcs import FSP_plots 
import numpy as np

#sort of working
extras = 20
num_BR = 30
num_mixed = 10
iters= 200000
time = 300

#test
extras = 0
num_BR = 30000
num_mixed = 20000
iters = 100000000
time = 36000
RL_iters = 1
check_freq = 1

opts = get_args.run()
iters = opts["num_lvls"]
game_obj = opts["game"]
#new test
#extras = 0
#Num_BR = 3000
#Num_mixed = 2000
#Iters = 10000000
#RL_iters = 1000
#Time = 300
#pol = np.array([[1/3,2/3],[0,1],[1,0],[0,1],[1/3,2/3],[1,0]])
#pol = np.ones((6,2))/2
#exact = learners.kuhn_exact_solver(pol,1)
#import pdb; pdb.set_trace()
#game_obj = leduc.leduc_int()
#game_obj = KP.Kuhn_Poker_int_io()

RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage, game_obj.num_actions[p],\
                game_obj.num_states[p], init_adv=0, extra_samples = extras, tol=1e-5)\
               for p in range(2)]
#RL_learners = [learners.fitted_Q_iteration(0, (game_obj.num_states[p], game_obj.num_actions[p])) for p in range(2)] 
SL_learners = [learners.count_based_SL((game_obj.num_states[p], game_obj.num_actions[p])) for p in range(2)]

agents = [learners.complete_learner(RL_learners[p], SL_learners[p], num_loops = RL_iters) for p in range(2)]

worker = FSP.FSP(game_obj, agents, max_iters=iters, max_time=time, m=num_BR, n=num_mixed, exploit_freq=check_freq)
pi, exploitability, data = worker.run_algo()

FSP_plots(exploitability, worker.est_exploit_freq, [pi], 'kuhn') 

filename="results/FSP"
np.savez(filename, pi=pi, exploit=exploitability, extra_data=data)
