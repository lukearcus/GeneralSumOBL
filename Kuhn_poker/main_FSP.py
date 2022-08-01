import FSP
import game
import matplotlib.pyplot as plt

KP_game = game.Kuhn_Poker_int_io()
worker = FSP.FSP(KP_game, max_iters=1000, m=40,n=0)
pi, exploitability, data = worker.run_algo()
plt.plot(exploitability)
plt.show()

import pdb; pdb.set_trace()
