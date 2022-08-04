from game import *
from players import *
import learners
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

def gen_belief(p1, p2):
    belief_1 = np.zeros([6,3])+1
    belief_2 = np.zeros([6,3])+1
    for i in range(10000):
        game.start_game()
        player=p1
        while not game.ended:
            player.observe(game.observe())
            game.action(player.action())
            if player == p1:
                player = p2
            else:
                player = p1
        rewards = game.end_game()
        c_1 = game.cards[0] - 1
        c_2 = game.cards[1] - 1
        for s in p1.state_hist:
            belief_1[s,c_2] += 1
        for s in p2.state_hist:
            belief_2[s,c_1] += 1
        p1.reset()
        p2.reset()
    belief_1 /= np.sum(belief_1,1,keepdims=True)
    belief_2 /= np.sum(belief_2,1,keepdims=True)
    return belief_1, belief_2


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

def play_obl_game(p1, p2, game):
    """
    This is a mess, fix at some point
    """
    game.start_game()
    player=p1
    while not game.ended:
        player.observe(game.observe())
        game.action(player.action())
        if player == p1:
            player = p2
        else:
            player = p1
    rewards = game.end_game()
    if game.curr_bets[0] == game.curr_bets[1]:
        if p1.belief is not None:
            for state in p1.state_hist: 
                probs = p1.belief[state]
                fict = np.argmax(np.random.multinomial(1, pvals = probs))
                p1.state = state
                act = p1.action()
                if state < 3:
                    if act == "bet":
                        p2.state = 2+fict
                        p2_act = p2.action()
                        if p2_act == "bet":
                            if game.cards[0] > fict:
                                p1.Q_update(state, p1.act, 2, -1)
                            else:
                                p1.Q_update(state, p1.act, -2, -1)
                        else:
                            p1.Q_update(state, p1.act, 1, -1)
                    else:
                        p2.state = fict
                        p2_act = p2.action()
                        if p2_act == "bet":
                            p1.Q_update(state, p1.act, 0, 3+state)
                        else:
                            if game.cards[0] > fict:
                                p1.Q_update(state, p1.act, 1, -1)
                            else:
                                p1.Q_update(state, p1.act, -1, -1)
                else:
                    if act == "fold":
                        p1.Q_update(state, p1.act, -1, -1)
                    else:
                        if game.cards[0] > fict:
                            p1.Q_update(state, p1.act, 2, -1)
                        else:
                            p1.Q_update(state, p1.act, -2, -1)


        else:
            p1.get_reward(rewards[0])
        if p2.belief is not None:
            probs_2 = p2.belief[p2.state]
            fict_s2 = np.argmax(np.random.multinomial(1, pvals = probs_2)) + 1
            if game.cards[1] > fict_s2:
                p2.get_reward(max(rewards))
            else:
                p2.get_reward(min(rewards))
        else:
            p2.get_reward(rewards[1])
    else:
        p1.get_reward(rewards[0])
        p2.get_reward(rewards[1])
    p1.reset()
    p2.reset()

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
