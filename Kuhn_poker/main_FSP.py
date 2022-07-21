import FSP
import game

KP_game = game.Kuhn_Poker_int_io()
worker = FSP.FSP(KP_game, max_iters=1000)
pi = worker.run_algo()
import pdb; pdb.set_trace()
