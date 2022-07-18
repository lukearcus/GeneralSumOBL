import random

class Kuhn_Poker:
    curr_player = 0
    curr_bets = [0,0]
    cards = [0,0]
    winner = 0
    ended = False

    def __init__(self):
        random.seed(1)

    def start_game(self):
        self.curr_bets = [1,1]
        cards_available = [1,2,3]
        random.shuffle(cards_available)
        self.cards = cards_available[:2]
        self.turn = 0
        self.curr_player = 0
        self.ended = False

    def observe(self):
        return self.cards[self.curr_player], self.curr_bets

    def action(self, act):
        if act=="bet":
            self.curr_bets[self.curr_player] +=1
        if act=="check":
            pass
        if act=="fold":
            self.winner = not self.curr_player
            self.ended = True
        if self.curr_bets[0] == self.curr_bets[1] and self.turn != 0:
            self.showdown()
        else:
            self.turn += 1
            self.curr_player = int(not self.curr_player)

    def showdown(self):
        self.winner = self.cards.index(max(self.cards))
        self.ended = True

    def end_game(self):
        loser = int(not self.winner)
        loss = -self.curr_bets[loser]
        winnings = self.curr_bets[loser]
        rewards = [0,0]
        rewards[self.winner] = winnings
        rewards[loser] = loss
        return rewards
