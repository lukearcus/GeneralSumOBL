from games.kuhn import *
from games.leduc import *
import agents.learners as learners
import sys
import time
import logging
log = logging.getLogger(__name__)


def run():
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
                    exploit_learner = learners.kuhn_exact_solver()
                elif sys.argv[game_ind+1] == "leduc":
                    game = leduc_int()
                    fict_game = leduc_fict()
                    exploit_learner = learners.actor_critic(learners.softmax, learners.value_advantage, \
                                            game.num_actions[0], game.num_states[0], tol=9999) 
                else:
                    print("Please enter a game choice")
                    return -1
            else:
                print("Please select a game")
                return(-1)
        else:
            game = Kuhn_Poker_int_io()
            fict_game = Fict_Kuhn_int()
            exploit_learner = learners.kuhn_exact_solver()
            
    else:
        num_lvls = 10
        game = Kuhn_Poker_int_io()
        fict_game = Fict_Kuhn_int()
        exploit_learner = learners.kuhn_exact_solver()
    if '--all_avg' in sys.argv or '-a' in sys.argv:
        averaged_bel = True
        averaged_pol = True
        learn_with_avg = True
    else:
        averaged_bel ='--avg_bel' in sys.argv or '-ab' in sys.argv
        averaged_pol ='--avg_pol' in sys.argv or '-ap' in sys.argv
        learn_with_avg = '--avg_learn' in sys.argv or '-al' in sys.argv
    if '--debug' in sys.argv:
        logging.basicConfig(level=logging.DEBUG,\
                format='%(relativeCreated)6d %(threadName)s %(message)s')
    elif '-v' in sys.argv or '--verbose' in sys.argv:
        logging.basicConfig(level=logging.INFO,\
                format='%(relativeCreated)6d %(threadName)s %(message)s')
    return num_lvls, game, fict_game, exploit_learner, averaged_bel, averaged_pol, learn_with_avg
