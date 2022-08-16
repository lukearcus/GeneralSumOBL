from Kuhn_poker.game import *
from agents.players import *
import agents.learners as learners
from UI.plot_funcs import plot_everything
from functions import play_game
import numpy as np
import sys

def main():
    game = Kuhn_Poker_int_io()
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
        averaged ='--avg' in sys.argv or '-a' in sys.argv
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
    avg_bels = []
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
        belief_hist.append(bels)
        if averaged:
            new_avg_bels = []
            for p_id, p in enumerate(players):
                total_bel = np.zeros_like(belief_hist[0][p_id])
                for i in range(lvl+1):
                    total_bel += belief_hist[i][p_id]
                avg_bel = total_bel / (lvl+1)
                p.belief = np.copy(avg_bel)
                new_avg_bels.append(avg_bel)
            avg_bels.append(new_avg_bels)
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
    
    if averaged:
        new_avg_bels = []
        for p_id, p in enumerate(players):
            total_bel = np.zeros_like(belief_hist[0][p_id])
            for i in range(lvl+1):
                total_bel += belief_hist[i][p_id]
            avg_bel = total_bel / (lvl+1)
            new_avg_bels.append(avg_bel)
        avg_bels.append(new_avg_bels)
    #pol_hist = pol_hist[-5:]
    #belief_hist = belief_hist[-5:]
    
    if averaged:
        plot_everything(pol_hist, avg_bels, "kuhn", reward_hist[-1])
    else:
        plot_everything(pol_hist, belief_hist, "kuhn", reward_hist[-1])
    
    import pdb; pdb.set_trace()
    return 0

if __name__=="__main__":
    main()
