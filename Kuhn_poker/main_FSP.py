import FSP
import game
import matplotlib.pyplot as plt
import learners

KP_game = game.Kuhn_Poker_int_io()

RL_learners = [learners.actor_critic(learners.softmax, learners.value_advantage, 2, 6, extra_samples = 0)\
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

worker = FSP.FSP(KP_game, agents, max_iters=20, m=30,n=0)
pi, exploitability, data = worker.run_algo()
plt.plot(exploitability)
plt.show()

import pdb; pdb.set_trace()
