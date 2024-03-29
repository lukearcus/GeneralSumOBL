from Kuhn_poker.game import *
from agents.players import *
import agents.learners as learners
from UI.plot_funcs import reward_smoothed
from functions import calc_exploitability
import matplotlib.pyplot as plt
import numpy as np

game = Kuhn_Poker_int_io()
games_per_lvl=100000
num_players = 2
learner = learners.actor_critic(learners.softmax, learners.value_advantage, 2, 6, init_lr=0.01, extra_samples = 0)
pol = [np.ones((6,2))/2 for i in range(2)]
#pol = [np.array([[0.2045, 0.795],[0.7105,0.2895],[0.6680,0.3320],[0.7231,0.2769],[0.5,0.5],[0.5,0.5]]),\
#       np.array([[0.6861, 0.3139],[0.4518,0.5482],[0.4385,0.5615],[0.1512,0.8488],[0.7143,0.2857],[0.5833,0.4167]])]
#pol = [np.array([[0.75,0.25],[0.75,0.25],[0.75,0.25],[0.5,0.5],[0.5,0.5],[0.5,0.5]]),\
#       np.array([[0.67,0.33],[0.69,0.31],[0.71,0.29],[0.19,0.81],[0.77,0.23],[0.79,0.21]])]
pol = [np.array([[2/3, 1/3],[2/3,1/3],[2/3,1/3],[1/3,2/3],[2/3,1/3],[2/3,1/3]]) for i in range(2)]
pol = [np.array([[0.816, 0.184],[0.811,0.189],[0.811,0.189],[0.375,0.625],[0.625,0.375],[0.625,0.376]]),\
       np.array([[0.53, 0.47],[0.771,0.229],[0.775,0.225],[0.159,0.841],[0.842,0.158],[0.838,0.162]])]

alpha=1/3
pol = [np.array([[alpha, 1-alpha],[0,1],[3*alpha,1-3*alpha],[0,1],[alpha+1/3,2/3-alpha],[1,0]]),\
       np.array([[1/3, 2/3],[0,1],[1,0],[0,1],[1/3,2/3],[1,0]])]

exploitability, pols, rewards = calc_exploitability(pol, game, learner, num_iters=100000) 

print(exploitability)
print(pols[0])
print(pols[1])
#print(V_1)
#print(V_2)
#fig = plt.figure()
#ax = fig.subplots()
#reward_smoothed(reward_hist, ax)
#
#plt.plot(change[0])
#plt.plot(change[1])
#plt.show()


import pdb; pdb.set_trace()
