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

    options = get_args.run() 
    num_lvls = options["num_lvls"]
    game_name = options["game_name"]
    game = options["game"]
    fict_game = options["fict_game"]
    exploit_learner = options["exploit_learner"]
    averaged_bel = options["avg_bel"]
    averaged_pol = options["avg_pol"]
    learn_with_avg = options["learn_w_avg"]
    learner_type = options["learner_type"]
    
    games_per_lvl=100000
    exploit_freq= 1
    
    num_players = 2
    if learner_type == "rl":
        RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage,\
                   game.num_actions[p], game.num_states[p], extra_samples = 0)\
                    for p in range(num_players)]
        players = [RL(RL_learners[p],p) for p in range(num_players)]
        if averaged_pol or learn_with_average:
            raise NotImplementedError
    elif learner_type == "obl":
        RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage,\
                   game.num_actions[p], game.num_states[p], extra_samples = 0)\
                    for p in range(num_players)]
        players = [OBL(RL_learners[p], p, fict_game) for p in range(num_players)]
    elif learner_type == "ot_rl":
        RL_learners = [[learners.actor_critic(learners.softmax, learners.value_advantage,\
                   game.num_actions[p], game.num_states[p], extra_samples = 0)\
                   for lvl in range(num_lvls)] for p in range(num_players)]
        players = [OT_RL(RL_learners[p], p, fict_game) for p in range(num_players)]
    fixed_players = [fixed_pol(players[p].opt_pol) for p in range(num_players)]
    
    for p in range(num_players):
        curr_player = players.pop(p)
        fixed_curr = fixed_players.pop(p)
        if curr_player.belief is not None:
            if learn_with_avg and learner_type == "obl":
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
        log.info("Level: " + str(lvl))
        pols = []
        bels = []
        for p in players:
            pols.append(p.opt_pol)
            if p.belief is not None:
                if learn_with_avg:
                    for p_id, other_p in enumerate(p.other_players):
                        if other_p != "me":
                            other_p.opt_pol = players[p_id].opt_pol
                if not averaged_bel:
                    p.belief_buff = []
                p.update_mem_and_bel()
                bels.append(np.copy(p.belief))
            else:
                bels.append(np.zeros((1,1)))
        pol_hist.append(pols)
        log.debug("Policies: " + str(pols))
        belief_hist.append(bels)
        log.debug("Beliefs: " + str(bels))
        if averaged_pol or learn_with_avg:
            new_avg_pols = []
            for p in players:
                if learner_type != "rl":
                    new_avg_pols.append(p.avg_pol)
            avg_pols.append(new_avg_pols)
            log.debug("Average polices: " + str(new_avg_pols))
        if lvl % exploit_freq == 0 and learner_type != "ot_rl":
            if averaged_pol:
                exploit, _, _, _ = calc_exploitability(new_avg_pols, game, exploit_learner)
            else:
                exploit, _, _, _ = calc_exploitability(pols, game, exploit_learner)
            exploitability.append(exploit)
            log.info("Exploitability: " + str(exploit))
        if learn_with_avg:
            for p_id, p in enumerate(players):
                for other_p_id, other_pol in enumerate(new_avg_pols):
                    if other_p_id != p_id:
                        p.other_players[other_p_id].opt_pol = other_pol
        for p in players:
            p.reset()
        play_to_convergence(players, game, tol=1e-7) 
        times.append(time.perf_counter()-tic)
    pols = []
    bels = []
    for p in players:
        pols.append(p.opt_pol)
        if p.belief is not None:
            if not averaged_bel:
                p.belief_buff = []
            p.update_mem_and_bel()
            bels.append(p.belief)
        else:
            bels.append(np.zeros((1,1)))
    pol_hist.append(pols)
    belief_hist.append(bels)
    
    #pol_hist = pol_hist[-5:]
    #belief_hist = belief_hist[-5:]
    if learner_type == 'ot_rl':
        pol_hist = []
        avg_pols = []
        for lvl in range(num_lvls):
            new_avg_pols = []
            pols = []
            for p in players:
                new_avg_pols.append(p.avg_pols[lvl])
                pols.append(p.pols[lvl])
            if averaged_pol:
                exploit, _, _, _ = calc_exploitability(new_avg_pols, game, exploit_learner)
                avg_pols.append(new_avg_pols)
            else:
                pol_hist.append(pols)
                exploit, _, _, _ = calc_exploitability(pols, game, exploit_learner)
            exploitability.append(exploit)
    else:
        if averaged_pol:
            new_avg_pols = []
            for p in players:
                new_avg_pols.append(p.avg_pol)
            avg_pols.append(new_avg_pols)
            exploit, _, _, _ = calc_exploitability(new_avg_pols, game, exploit_learner)
        else:
            exploit, _, _, _ = calc_exploitability(pols, game, exploit_learner)
        exploitability.append(exploit)

    if averaged_pol:
        pol_plot = avg_pols
    else:
        pol_plot = pol_hist
    bel_plot = belief_hist
    plot_everything(pol_plot, bel_plot, "kuhn", reward_hist[-1], exploitability)
    filename="results/" + game_name +  "_" + learner_type + "_" + str(num_lvls) + "lvls"
    np.savez(filename, pols=pol_plot, bels=bel_plot, exploit=exploitability, rewards=reward_hist)
    return 0

if __name__=="__main__":
    main()
