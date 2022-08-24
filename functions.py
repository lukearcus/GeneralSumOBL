from agents.players import RL, fixed_pol
import numpy as np

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
        player.wipe_mem()
    return reward

def calc_exploitability(pol, game, learner, num_iters=100000, num_exploit_iters = 1000):
    players = [RL(learner,0), fixed_pol(pol[1])]
    
    reward_hist = [[],[]]
    
    change = [[], []]
    p_avg_exploitability = [[],[]]
    exploit_rewards = [[],[]]

    for i in range(num_iters):
        old_pol = np.copy(players[0].opt_pol)
        reward_hist[0].append(float(play_game(players, game)))
        change[0].append(np.linalg.norm(players[0].opt_pol-old_pol))
    
    new_pols = []
    new_pols.append(players[0].opt_pol)
    
    players = [fixed_pol(new_pols[0]), fixed_pol(pol[1])]
    
    for i in range(num_exploit_iters):
        exploit_rewards[0].append(float(play_game(players, game)))

    p_avg_exploitability[0] = sum(exploit_rewards[0])/len(exploit_rewards[0])
    V_1 = learner.advantage_func.V
    
    learner.reset()
    learner.wipe_memory()

    players = [fixed_pol(pol[0]), RL(learner,1)] 
    
    for i in range(num_iters):
        old_pol = np.copy(players[1].opt_pol)
        reward_hist[1].append(-float(play_game(players, game)))
        change[1].append(np.linalg.norm(players[1].opt_pol-old_pol))
    
    new_pols.append(players[1].opt_pol)
    players = [fixed_pol(pol[0]), fixed_pol(new_pols[1])]
    
    for i in range(num_exploit_iters):
        exploit_rewards[1].append(-float(play_game(players, game)))

    p_avg_exploitability[1] = sum(exploit_rewards[1])/len(exploit_rewards[1])
    
    V_2 = learner.advantage_func.V
    
    avg_exploitability = sum(p_avg_exploitability)
    learner.reset()
    learner.wipe_memory()

    #import pdb; pdb.set_trace()
    return avg_exploitability, new_pols, reward_hist, (V_1, V_2)
