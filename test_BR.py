from Kuhn_poker.game import *
from agents.players import *
import agents.learners as learners
from UI.plot_funcs import plot_everything
from functions import play_game

game = Kuhn_Poker_int_io()
games_per_lvl=100000
num_players = 2
RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage, 2, 6, extra_samples = 0)\
               for p in range(num_players)]
fict_game = Fict_Kuhn_int()
pol = [np.ones((6,2))/2 for i in range(2)]
#pol = [np.array([[0.2045, 0.795],[0.7105,0.2895],[0.6680,0.3320],[0.7231,0.2769],[0.5,0.5],[0.5,0.5]]),\
#       np.array([[0.6861, 0.3139],[0.4518,0.5482],[0.4385,0.5615],[0.1512,0.8488],[0.7143,0.2857],[0.5833,0.4167]])]
#pol = [np.array([[0.75,0.25],[0.75,0.25],[0.75,0.25],[0.5,0.5],[0.5,0.5],[0.5,0.5]]),\
#       np.array([[0.67,0.33],[0.69,0.31],[0.71,0.29],[0.19,0.81],[0.77,0.23],[0.79,0.21]])]
pol = [np.array([[2/3, 1/3],[2/3,1/3],[2/3,1/3],[1/3,2/3],[2/3,1/3],[2/3,1/3]]) for i in range(2)]
pol = [np.array([[0.816, 0.184],[0.811,0.189],[0.811,0.189],[0.375,0.625],[0.625,0.375],[0.625,0.376]]),\
       np.array([[0.53, 0.47],[0.771,0.229],[0.775,0.225],[0.159,0.841],[0.842,0.158],[0.838,0.162]])]

alpha=0.0
pol = [np.array([[alpha, 1-alpha],[0,1],[3*alpha,1-3*alpha],[0,1],[alpha+1/3,2/3-alpha],[1,0]]),\
       np.array([[1/3, 2/3],[0,1],[1,0],[0,1],[1/3,2/3],[1,0]])]

players = [RL(RL_learners[0],0), fixed_pol(pol[1])]

reward_hist = []

for i in range(games_per_lvl):
    reward_hist.append(float(play_game(players, game)))

R = reward_hist[-100:]
pols = []
pols.append(players[0].opt_pol)
V_1 = players[0].learner.advantage_func.V

players = [fixed_pol(pol[0]), RL(RL_learners[1],1)] 

for i in range(games_per_lvl):
    reward_hist.append(-float(play_game(players, game)))

R += reward_hist[-100:]
pols.append(players[1].opt_pol)
V_2 = players[1].learner.advantage_func.V

print(sum(R)/200)
print(pols[0])
print(pols[1])
print(V_1)
print(V_2)

import pdb; pdb.set_trace()
