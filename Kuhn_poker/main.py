from game import *
from players import *
import learners
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


def play_game(players, game):
    game.start_game()
    while not game.ended:
        player = players[game.curr_player]
        player.observe(game.observe())
        game.action(player.action())
    for i in players:
        player = players[game.curr_player]
        player.observe(game.observe())
        game.action(None)
    reward = players[0].r
    for player in players:
        player.reset()
    return reward

game = Kuhn_Poker_int_io()
num_lvls = 1
games_per_lvl=100000
num_players = 2
RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage, 2, 6, extra_samples = 0)\
               for p in range(num_players)]
fict_game = Fict_Kuhn_int()
#players = [RL(RL_learners[p],p) for p in range(num_players)]
pol = [np.ones((6,2))/2 for i in range(2)]
#pol = [np.array([[0.2045, 0.795],[0.7105,0.2895],[0.6680,0.3320],[0.7231,0.2769],[0.5,0.5],[0.5,0.5]]),\
#       np.array([[0.6861, 0.3139],[0.4518,0.5482],[0.4385,0.5615],[0.1512,0.8488],[0.7143,0.2857],[0.5833,0.4167]])]
pol = [np.array([[0.75,0.25],[0.75,0.25],[0.75,0.25],[0.5,0.5],[0.5,0.5],[0.5,0.5]]),\
       np.array([[0.67,0.33],[0.69,0.31],[0.71,0.29],[0.19,0.81],[0.77,0.23],[0.79,0.21]])]

#players = [fixed_pol(pol[0]), RL(RL_learners[1],1)] 
players = [RL(RL_learners[0],0), fixed_pol(pol[1])]
#players = [OBL(RL_learners[p], p, fict_game) for p in range(num_players)]
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

#reward_smoothed = gaussian_filter1d(reward_hist, sigma=50)
#plt.plot(reward_smoothed)
plt.show()

fig, axs = plt.subplots(num_lvls+1,4)
for level in range(num_lvls+1):
    axs[level,2].imshow(belief_hist[level][0])
    axs[level,3].imshow(belief_hist[level][1])
    x_label_list = ["","bet","check"]
    y_label_list = ["","1 low", "2 low", "3 low", "1 high", "2 high", "3 high"]
    axs[level,0].imshow(pol_hist[level][0])
    axs[level,1].imshow(pol_hist[level][1])
    axs[level,0].set_xticklabels(x_label_list)
    axs[level,0].set_yticklabels(y_label_list)
    axs[level,1].set_xticklabels(x_label_list)
    axs[level,1].set_yticklabels(y_label_list)

plt.show()

import pdb; pdb.set_trace()
