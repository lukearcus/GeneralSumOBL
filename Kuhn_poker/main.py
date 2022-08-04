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
    reward = players[0].buffer[-1]["r"]
    for player in players:
        player.reset()
    return reward

game = Kuhn_Poker_int_io()
num_lvls = 4
games_per_lvl=1000
num_players = 2
RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage, 2, 6, extra_samples = 0)\
               for p in range(num_players)]
fict_game = Fict_Kuhn_int()
#players = [RL(RL_learners[p],p) for p in range(num_players)]

players = [OBL(RL_learners[p], p, fict_game) for p in range(num_players)]
for p in range(num_players):
    curr_player = players.pop(p)
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
        p.update_belief()
        bels.append(p.belief)
    pol_hist.append(pols)
    belief_hist.append(bels)
    for i in range(games_per_lvl):
        reward_hist[lvl][i] = float(play_game(players, game))
pols = []
bels = []
for p in players:
    pols.append(p.opt_pol)
    p.update_belief()
    bels.append(p.belief)
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
