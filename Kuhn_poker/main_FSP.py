import FSP
import game
import matplotlib.pyplot as plt

KP_game = game.Kuhn_Poker_int_io()
worker = FSP.FSP(KP_game, max_iters=10000, m=2,n=1)
pi, exploitability, data = worker.run_algo()
plt.plot(exploitability)
plt.show()

import pdb; pdb.set_trace()
