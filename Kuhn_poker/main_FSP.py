import FSP
import game
import matplotlib.pyplot as plt
import learners

#sort of working
extras = 20
num_BR = 30
num_mixed = 10
iters= 200000
time = 60

#test
#extras = 2
#num_BR = 4
#num_mixed = 0
#iters = 100000
#time = 60

KP_game = game.Kuhn_Poker_int_io()

RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage, 2, 6, init_adv=-2, extra_samples = extras)\
               for p in range(2)]
SL_learners = [learners.count_based_SL((6,2)) for p in range(2)]


#import random
#import numpy as np
#hist = []
#pol = RL_learners[0].opt_pol
#for j in range(10):
#    for i in range(100):
#        s = random.randint(0,5)
#        a = np.argmax(np.random.multinomial(1, pvals = pol[s,:]))
#        if s < 3:
#            if a == 1:
#                s_prime = s + 3
#                r = 0
#            else:
#                s_prime = -1
#                r = 1
#        else:
#            if a == 1:
#                s_prime = -1
#                r = -1
#            else:
#                s_prime = -1
#                r = 2
#        hist.append(([{"s":s, "a":a, "r":r, "s'":s_prime}], None, None))
#    RL_learners[0].update_memory(hist)
#    pol = RL_learners[0].learn()
#    print(pol)
#import pdb; pdb.set_trace()

agents = [learners.complete_learner(RL_learners[p], SL_learners[p]) for p in range(2)]

worker = FSP.FSP(KP_game, agents, max_iters=iters, max_time=time, m=num_BR, n=num_mixed)
pi, exploitability, data = worker.run_algo()
plt.plot(exploitability)
plt.show()

import pdb; pdb.set_trace()
