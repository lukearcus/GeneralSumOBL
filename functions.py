
def play_game(players, game):
    game.start_game()
    while not game.ended:
        player = players[game.curr_player]
        player.observe(game.observe())
        game.action(player.action())
    for i in players:
        player = players[game.curr_player]
        player.observe(game.observe())
        game.action(None)
    reward = players[0].r
    for player in players:
        player.wipe_mem()
    return reward

