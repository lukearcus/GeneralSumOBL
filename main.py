from Kuhn_poker.game import *
from agents.players import *
import agents.learners as learners
from scipy.ndimage.filters import gaussian_filter1d
from UI.plot_funcs import plot_everything
from functions import play_game

game = Kuhn_Poker_int_io()
num_lvls = 1
games_per_lvl=100000
num_players = 2
RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage, 2, 6, extra_samples = 0)\
               for p in range(num_players)]
fict_game = Fict_Kuhn_int()

#players = [RL(RL_learners[p],p) for p in range(num_players)]
players = [OBL(RL_learners[p], p, fict_game) for p in range(num_players)]

for p in range(num_players):
    curr_player = players.pop(p)
    if curr_player.belief is not None:
        curr_player.set_other_players(players)
    players.insert(p, curr_player)

reward_hist = [[0 for i in range(games_per_lvl)] for lvl in range(num_lvls)]
pol_hist = []
belief_hist = []
for lvl in range(num_lvls):
    pols = []
    bels = []
    for p in players:
        pols.append(p.opt_pol)
        if p.belief is not None:
            p.update_belief()
            bels.append(p.belief)
        else:
            bels.append(np.zeros((1,1)))
    pol_hist.append(pols)
    belief_hist.append(bels)
    for p in players:
        p.reset()
    for i in range(games_per_lvl):
        reward_hist[lvl][i] = float(play_game(players, game))
pols = []
bels = []
for p in players:
    pols.append(p.opt_pol)
    if p.belief is not None:
        p.update_belief()
        bels.append(p.belief)
    else:
        bels.append(np.zeros((1,1)))
pol_hist.append(pols)
belief_hist.append(bels)

#pol_hist = pol_hist[-5:]
#belief_hist = belief_hist[-5:]

plot_everything(pol_hist, belief_hist, "kuhn")

import pdb; pdb.set_trace()
