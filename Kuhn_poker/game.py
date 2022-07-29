import random
from itertools import product

class Kuhn_Poker:
    curr_player = 0
    curr_bets = [0,0]
    cards = [0,0]
    winner = 0
    ended = False
    num_players = 2

    def __init__(self, _num_players=2):
        random.seed(1)
        self.num_players = _num_players
        self.num_states = [(self.num_players+1)*(2**(self.num_players-1)) for i in  range(self.num_players)]
        self.num_actions = [2 for i in range(self.num_players)]

    def start_game(self):
        self.curr_bets = [1 for i in range(self.num_players)]
        cards_available = [ i+1 for i in range(self.num_players+1)]
        random.shuffle(cards_available)
        self.cards = cards_available[:self.num_players]
        self.curr_player = 0
        self.rewards = [0 for i in range(self.num_players)]
        self.ended = False
        self.folded = [False for i in range(self.num_players)]
        self.checked = [False for i in range(self.num_players)]
        self.betted = [False for i in range(self.num_players)]

    def observe(self):
        if not self.ended:
            return self.cards[self.curr_player], self.curr_bets, 0
        else: 
            return -1, None, self.rewards[self.curr_player]

    def action(self, act):
        if not self.ended:
            if act=="bet":
                self.curr_bets[self.curr_player] +=1
                self.betted[self.curr_player] = True
            if act=="check":
                if any(self.betted):
                    self.folded[self.curr_player] = True
                else:
                    self.checked[self.curr_player] = True
            if act=="fold":
                self.folded[self.curr_player] = True
            if all(self.checked):
                self.find_winner()
            elif all(p[0] or p[1] for p in zip(self.betted, self.folded)):
                self.find_winner()
        self.curr_player += 1
        if self.curr_player == self.num_players:
            self.curr_player = 0

    def find_winner(self):
        if any(self.betted):
            bets = True
        else:
            bets = False
        if bets:
            valid_cards = [self.cards[i] for i in range(self.num_players) if self.betted[i]]
        else:
            valid_cards = self.cards
        best_card = max(valid_cards)
        self.winner = self.cards.index(best_card)
        self.ended = True
        losses = [-bet for bet in self.curr_bets]
        winnings = sum(self.curr_bets)
        self.rewards = losses
        self.rewards[self.winner] += winnings

class Kuhn_Poker_int_io(Kuhn_Poker):
    
    def observe(self):    
        card, game_pot, reward = super().observe()
        if card != -1:
            pot = game_pot.copy()
            poss_pots = list(product([1,2],repeat=self.num_players-1))
            pot.pop(self.curr_player)
            pot_ind = poss_pots.index(tuple(pot))
            return pot_ind*(self.num_players+1)+card-1, reward
        else:
            return -1, reward

    def action(self, act):
        if act == 0:
            super().action("bet")
        else:
            super().action("check")
