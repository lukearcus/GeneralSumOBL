from agents.players import *
import agents.learners as learners
from UI.plot_funcs import plot_everything
import UI.get_args as get_args
from functions import *
import numpy as np
import sys
import time
import logging
from multiprocessing import Pool
import OBL
import FSP
log = logging.getLogger(__name__)
NUM_LOOPS=1 # Change this to run multiple seeds (in parallel)


def main():
    options = get_args.run() 
    if options["mode"] == "obl":
        with Pool(NUM_LOOPS) as p:
            all_res = p.map(OBL.run, [options for i in range(NUM_LOOPS)])
        #plot_everything(pol_plot, bel_plot, "kuhn", reward_hist[-1], exploitability)
        log.info("Saving results...")
        filename="results/" + options["game_name"] +  "_" + options["learner_type"] + "_"+str(NUM_LOOPS)+"_loops"
        np.savez(filename, res=all_res)
        #np.savez(filename, pols=pol_plot, bels=bel_plot, exploit=exploitability, rewards=reward_hist, times=times)
        log.info("Completed")
        return 0
    else:
        with Pool(NUM_LOOPS) as p:
            all_res = p.map(FSP.run, [options for i in range(NUM_LOOPS)])
        log.info("Saving results...")
        filename="results/FSP_" + options["game_name"] +  "_"+str(NUM_LOOPS)+"_loops"
        np.savez(filename, res=all_res)
        #np.savez(filename, pols=pol_plot, bels=bel_plot, exploit=exploitability, rewards=reward_hist, times=times)
        log.info("Completed")
        return 0

if __name__=="__main__":
    main()
