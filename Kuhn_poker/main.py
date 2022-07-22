from game import Kuhn_Poker
from players import *
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


def play_game(p1, p2, game):
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
    p1.get_rewards(rewards[0])
    p2.get_rewards(rewards[1])
    p1.reset()
    p2.reset()
    return rewards

def play_obl_game(p1, p2, game):
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
        if p1.belief is not None and len(p1.state_hist) == 2:
            probs_1 = p1.belief[p1.state]
            fict_s1 = np.argmax(np.random.multinomial(1, pvals = probs_1))
            if game.cards[0] > fict_s1:
                p1.get_reward(max(rewards))
            else:
                p1.get_reward(min(rewards))
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

#p2 = human()
game = Kuhn_Poker()
num_lvls =1
beliefs = []

p1 = human()
p2 = human()
#p1 = vanilla_rl(0,6,2, learning_rate=0, T=1)
#p1.Q_mat[2,0] = 10
#p1.Q_mat[0,1] = 10
#p1 = vanilla_rl(0,6,2,T=1)
#p2 = vanilla_rl(0,6,2, T=1)
#p1 = OBL(0,6,2, T=1)
#p2 = OBL(0,6,2, T=1)

Q_mats = [(np.copy(p1.Q_mat),np.copy(p2.Q_mat))]
Q_change = []
for lvl in range(num_lvls):
    if p1.belief is not None or p2.belief is not None:
        b1,b2 = gen_belief(p1,p2)
        beliefs.append((b1,b2))
    if p1.belief is not None:
        p1.set_belief(b1)
    if p2.belief is not None:
        p2.set_belief(b2)
    q1_old = np.copy(p1.Q_mat)
    q2_old = np.copy(p2.Q_mat)
    for i in range(1000):
        play_obl_game(p1,p2,game)
        Q_change.append((np.linalg.norm(q1_old-p1.Q_mat), np.linalg.norm(q2_old-p2.Q_mat)))
        q1_old = np.copy(p1.Q_mat)
        q2_old = np.copy(p2.Q_mat)
        p1.eps = 1-i/1000
        p2.eps = 1-i/1000
    Q_mats.append((np.copy(p1.Q_mat),np.copy(p2.Q_mat)))
    p1.reset_Q()
    p2.reset_Q()
b1,b2 = gen_belief(p1,p2)
beliefs.append((b1,b2))
#reward_hist = []
#for i in range(1000):
#   reward_hist.append(play_game())

#p2 = human()
#for i in range(1000):
#    game.start_game()
#    player=p1
#    while not game.ended:
#        player.observe(game.observe())
#        game.action(player.action())
#        if player == p1:
#            player = p2
#        else:
#            player = p1
#    rewards = game.end_game()
#    reward_hist.append(rewards[0])
#    p1.get_rewards(rewards[0])
#    p2.get_rewards(rewards[1])
#    p1.reset()
#    p2.reset()

#reward_smoothed = gaussian_filter1d(reward_hist, sigma=1)
#plt.plot(reward_smoothed)
#plt.show()

plt.plot(Q_change)
plt.show()
fig, axs = plt.subplots(num_lvls+1,4)
for level in range(num_lvls+1):
    axs[level,2].imshow(beliefs[level][0])
    axs[level,3].imshow(beliefs[level][1])
    x_label_list = ["","bet/call","check/fold"]
    y_label_list = ["","no raise 1", "no raise 2", "no raise 3", "raised 1", "raised 2", "raised 3"]
    axs[level,0].imshow(Q_mats[level][0])
    axs[level,1].imshow(Q_mats[level][1])
    axs[level,0].set_xticklabels(x_label_list)
    axs[level,0].set_yticklabels(y_label_list)
    axs[level,1].set_xticklabels(x_label_list)
    axs[level,1].set_yticklabels(y_label_list)

plt.show()

f, (ax1, ax2) = plt.subplots(1,2)
