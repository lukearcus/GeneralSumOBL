from agents.players import RL, fixed_pol
from agents import learners
import numpy as np
import logging

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

def play_to_convergence(players, game, max_iters=1000000, tol=1e-5):
    old_pol = [None for p in players]
    converged_itt = 0
    for i in range(max_iters):
        converged_itt += 1
        for k, p in enumerate(players):
            old_pol[k] = np.copy(p.opt_pol)
        play_game(players, game)
        converged = True
        for j, p in enumerate(players):
            pol_diff = np.linalg.norm(p.opt_pol-old_pol[j],ord=np.inf)
            logging.debug("Iteration "+str(i) + " diff " + str(pol_diff))
            converged = converged and pol_diff <= tol
            if not converged:
                break
        if converged and i>100:
            break
    if not converged:
        logging.warning("Did not Converge")
        return -1
    else:
        logging.info("Converged after " + str(converged_itt) + " iterations")
        return 0

def calc_exploitability(pol, game, learner, num_iters=100000, num_exploit_iters = 10000, tol=1e-10, exploit_tol = 1e-4):
    new_pols = []
    p_avg_exploitability = [0,0]
    exploit_rewards = [[],[]]
    if isinstance(learner, learners.kuhn_exact_solver):
        new_pols.append(learner.calc_opt(pol[1],1))
        reward_hist = None
        V_1 = None
    else:
        players = [RL(learner,0), fixed_pol(pol[1])]
        
        reward_hist = [[],[]]
        
        change = [[], []]
        i = 0
        while True:
            old_pol = np.copy(players[0].opt_pol)
            reward_hist[0].append(float(play_game(players, game)))
            change[0].append(np.linalg.norm(players[0].opt_pol-old_pol, ord=np.inf))
            i += 1
            if i == num_iters:
                break
            elif i>100 and change[0][-1] <= tol:
                break
        converged_pol = players[0].opt_pol + np.random.random(players[0].opt_pol.shape)/1000
        opt_deterministic = np.array(np.invert(np.array(converged_pol-\
                                     np.max(converged_pol, axis=1,keepdims=True),\
                                     dtype=bool)),\
                                     dtype=float)
        new_pols.append(opt_deterministic)
        #new_pols.append(players[0].opt_pol)
        V_1 = learner.advantage_func.V
    
    players = [fixed_pol(new_pols[0]), fixed_pol(pol[1])]
    i = 0
    while True:
        old_exploitability = p_avg_exploitability[0]
        exploit_rewards[0].append(float(play_game(players, game)))
        p_avg_exploitability[0] = sum(exploit_rewards[0])/len(exploit_rewards[0])
        i += 1
        if i == num_exploit_iters:
            break 
        elif i>100 and np.abs(old_exploitability - p_avg_exploitability[0]) < exploit_tol:
            break

    p_avg_exploitability[0] = sum(exploit_rewards[0])/len(exploit_rewards[0])
    
    if isinstance(learner, learners.kuhn_exact_solver):
        new_pols.append(learner.calc_opt(pol[0],2))
        V_2 = None
    else:
    
        learner.reset()
        learner.wipe_memory()

        players = [fixed_pol(pol[0]), RL(learner,1)] 
   
        i = 0
        while True:
            old_pol = np.copy(players[1].opt_pol)
            reward_hist[1].append(-float(play_game(players, game)))
            change[1].append(np.linalg.norm(players[1].opt_pol-old_pol, ord=np.inf))
            i += 1
            if i == num_iters:
                break
            elif i>100 and change[1][-1] <= tol:
                break
        
        V_2 = learner.advantage_func.V
        converged_pol = players[1].opt_pol + np.random.random(players[1].opt_pol.shape)/1000
        opt_deterministic = np.array(np.invert(np.array(converged_pol-\
                                     np.max(converged_pol, axis=1,keepdims=True),\
                                     dtype=bool)),\
                                     dtype=float)
        new_pols.append(opt_deterministic)
        #new_pols.append(players[1].opt_pol)
        learner.reset()
        learner.wipe_memory()
    players = [fixed_pol(pol[0]), fixed_pol(new_pols[1])]
    
    i = 0
    while True:
        old_exploitability = p_avg_exploitability[1]
        exploit_rewards[1].append(-float(play_game(players, game)))
        p_avg_exploitability[1] = sum(exploit_rewards[1])/len(exploit_rewards[1])
        i+= 1
        if i == num_exploit_iters:
            break
        elif i > 100 and np.abs(old_exploitability - p_avg_exploitability[1]) < exploit_tol:
            break

    p_avg_exploitability[1] = sum(exploit_rewards[1])/len(exploit_rewards[1])
    
    
    avg_exploitability = sum(p_avg_exploitability)

    #import pdb; pdb.set_trace()
    return avg_exploitability, new_pols, reward_hist, (V_1, V_2)
