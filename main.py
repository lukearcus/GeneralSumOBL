from agents.players import *
import agents.learners as learners
from UI.plot_funcs import plot_everything
import UI.get_args as get_args
from functions import *
import numpy as np
import sys
import time
import logging
log = logging.getLogger(__name__)

def main():

    num_lvls, game, fict_game, exploit_learner, averaged_bel, averaged_pol, learn_with_avg = get_args.run() 

    games_per_lvl=100000
    exploit_freq= 1
    
    num_players = 2
    RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage,\
                   game.num_actions[p], game.num_states[p], extra_samples = 0)\
                   for p in range(num_players)]
    #players = [RL(RL_learners[p],p) for p in range(num_players)]
    players = [OBL(RL_learners[p], p, fict_game) for p in range(num_players)]
    fixed_players = [fixed_pol(players[p].opt_pol) for p in range(num_players)]
    
    for p in range(num_players):
        curr_player = players.pop(p)
        fixed_curr = fixed_players.pop(p)
        if curr_player.belief is not None:
            if learn_with_avg:
                curr_player.set_other_players(fixed_players)
            else:
                curr_player.set_other_players(players)
        fixed_players.insert(p, fixed_curr)
        players.insert(p, curr_player)
    
    reward_hist = [[0 for i in range(games_per_lvl)] for lvl in range(num_lvls)]
    pol_hist = []
    belief_hist = []
    avg_pols = []
    avg_bels = []
    exploitability = []
    times = []
    tic = time.perf_counter()
    for lvl in range(num_lvls):
        pols = []
        bels = []
        for p in players:
            pols.append(p.opt_pol)
            if p.belief is not None:
                if learn_with_avg:
                    for p_id, other_p in enumerate(p.other_players):
                        if other_p != "me":
                            other_p.opt_pol = players[p_id].opt_pol
                p.update_belief()
                bels.append(np.copy(p.belief))
            else:
                bels.append(np.zeros((1,1)))
        pol_hist.append(pols)
        log.debug("Policies at lvl "+str(lvl) + ": " + str(pols))
        belief_hist.append(bels)
        log.debug("Beliefs at lvl "+str(lvl) + ": " + str(bels))
        if averaged_bel:
            new_avg_bels = []
            for p_id, p in enumerate(players):
                total_bel = np.zeros_like(belief_hist[0][p_id])
                for i in range(lvl+1):
                    total_bel += belief_hist[i][p_id]
                avg_bel = total_bel / (lvl+1)
                p.belief = np.copy(avg_bel)
                new_avg_bels.append(avg_bel)
            avg_bels.append(new_avg_bels)
            log.debug("Average beliefs at lvl "+str(lvl) + ": " + str(new_avg_bels))
        if averaged_pol or learn_with_avg:
            new_avg_pols = []
            for p_id, p in enumerate(players):
                total_pol = np.zeros_like(pol_hist[0][p_id])
                for i in range(lvl+1):
                    total_pol += pol_hist[i][p_id]
                avg_pol = total_pol / (lvl+1)
                new_avg_pols.append(avg_pol)
            avg_pols.append(new_avg_pols)
            log.debug("Average polices at lvl "+str(lvl) + ": " + str(new_avg_pols))
        if lvl % exploit_freq == 0:
            if averaged_pol:
                exploit, _, _, _ = calc_exploitability(new_avg_pols, game, exploit_learner)
            else:
                exploit, _, _, _ = calc_exploitability(pols, game, exploit_learner)
            exploitability.append(exploit)
            log.info("Exploitability at lvl " + str(lvl) + ": " + str(exploit))
        if learn_with_avg:
            for p_id, p in enumerate(players):
                for other_p_id, other_pol in enumerate(new_avg_pols):
                    if other_p_id != p_id:
                        p.other_players[other_p_id].opt_pol = other_pol
        for p in players:
            p.reset()
        play_to_convergence(players, game, tol=1e-7) 
        #for i in range(games_per_lvl):
        #    reward_hist[lvl][i] = float(play_game(players, game))
        times.append(time.perf_counter()-tic)
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
    
    if averaged_bel:
        new_avg_bels = []
        for p_id, p in enumerate(players):
            total_bel = np.zeros_like(belief_hist[0][p_id])
            for i in range(lvl+1):
                total_bel += belief_hist[i][p_id]
            avg_bel = total_bel / (lvl+1)
            new_avg_bels.append(avg_bel)
        avg_bels.append(new_avg_bels)
    if averaged_pol:
        new_avg_pols = []
        for p_id, p in enumerate(players):
            total_pol = np.zeros_like(pol_hist[0][p_id])
            for i in range(lvl+1):
                total_pol += pol_hist[i][p_id]
            avg_pol = total_pol / (lvl+1)
            new_avg_pols.append(avg_pol)
        avg_pols.append(new_avg_pols)
        exploit, _, _, _ = calc_exploitability(new_avg_pols, game, exploit_learner)
    else:
        exploit, _, _, _ = calc_exploitability(pols, game, exploit_learner)
    exploitability.append(exploit)
    #pol_hist = pol_hist[-5:]
    #belief_hist = belief_hist[-5:]
    
    if averaged_pol:
        pol_plot = avg_pols
    else:
        pol_plot = pol_hist
    if averaged_bel:
        bel_plot = avg_bels
    else:
        bel_plot = belief_hist
    plot_everything(pol_plot, bel_plot, "kuhn", reward_hist[-1], exploitability)
    filename="results/OBL_all_average"
    np.savez(filename, pols=pol_plot, bels=bel_plot, explot=exploitability, rewards=reward_hist)
    return 0

if __name__=="__main__":
    main()
