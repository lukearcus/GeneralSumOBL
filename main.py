from games.kuhn import *
from games.leduc import *
from agents.players import *
import agents.learners as learners
from UI.plot_funcs import plot_everything
from functions import *
import numpy as np
import sys
import time
import logging
log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
    if len(sys.argv) > 1:
        if '--lvls' in sys.argv:
            level_ind = sys.argv.index('--lvls')
            if len(sys.argv) > level_ind:
                try:
                    num_lvls = int(sys.argv[level_ind+1])
                except TypeError:
                    print("Please enter a numerical value for number of levels")
                    return -1
            else:
                print("Please enter number of levels")
                return(-1)
        else:
            num_lvls = 10
        if '--game' in sys.argv:
            game_ind = sys.argv.index('--game')
            if len(sys.argv) > game_ind:
                if sys.argv[game_ind+1] == "kuhn":
                    game = Kuhn_Poker_int_io()
                    fict_game = Fict_Kuhn_int()
                elif sys.argv[game_ind+1] == "leduc":
                    game = leduc_int()
                    fict_game = leduc_fict() 
                else:
                    print("Please enter a game choice")
                    return -1
            else:
                print("Please select a game")
                return(-1)
        else:
            game = Kuhn_Poker_int_io()
            fict_game = Fict_Kuhn_int()
            
    else:
        num_lvls = 10
        game = Kuhn_Poker_int_io()
        fict_game = Fict_Kuhn_int()
    if '--all_avg' in sys.argv or '-a' in sys.argv:
        averaged_bel = True
        averaged_pol = True
        learn_with_avg = True
    else:
        averaged_bel ='--avg_bel' in sys.argv or '-ab' in sys.argv
        averaged_pol ='--avg_pol' in sys.argv or '-ap' in sys.argv
        learn_with_avg = '--avg_learn' in sys.argv or '-al' in sys.argv
    games_per_lvl=100000
    exploit_freq= 1
    
    num_players = 2
    RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage,\
                   game.num_actions[p], game.num_states[p], extra_samples = 0)\
                   for p in range(num_players)]
    exploit_learner = learners.actor_critic(learners.softmax, learners.value_advantage, \
                                            game.num_actions[0], game.num_states[0], tol=9999) 
    solver = learners.kuhn_exact_solver()
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
                true_exploit, true_br_pols, _, _ = calc_exploitability(new_avg_pols, game, solver,\
                                                                            num_iters = -1, num_exploit_iters=-1)
            else:
                exploit, _, _, _ = calc_exploitability(pols, game, exploit_learner)
                true_exploit, true_br_pols, _, _ = calc_exploitability(pols, game, solver,\
                                                                            num_iters = -1, num_exploit_iters=-1)
            exploitability.append(true_exploit)
            log.info("True exploitability at lvl " + str(lvl) + ": " + str(true_exploit))
            log.info("Esimated exploitability at lvl " + str(lvl) + ": " + str(exploit))
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
    filename="results/OBL"
    np.savez(filename, pols=pol_plot, bels=bel_plot, explot=exploitability, rewards=reward_hist)
    return 0

if __name__=="__main__":
    main()
