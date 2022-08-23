import random
from itertools import product
from games.base import base

class Kuhn_Poker(base):
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
                self.end_game()
            elif all(p[0] or p[1] for p in zip(self.betted, self.folded)):
                self.end_game()
        self.curr_player += 1
        if self.curr_player == self.num_players:
            self.curr_player = 0

    def end_game(self):
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
    
    def __init__(self):
        super().__init__()
        self.poss_pots = list(product([1,2],repeat=self.num_players-1))
    
    def observe(self):    
        card, game_pot, reward = super().observe()
        if card != -1:
            pot = game_pot.copy()
            pot.pop(self.curr_player)
            pot_ind = self.poss_pots.index(tuple(pot))
            return pot_ind*(self.num_players+1)+card-1, reward
        else:
            return -1, reward

    def action(self, act):
        if act == 0:
            super().action("bet")
        else:
            super().action("check")

class Fict_Kuhn_int(Kuhn_Poker_int_io):

    def __init__(self):
        super().__init__()
        self.poss_hidden = list(product(list(range(1,self.num_players+2)), \
                                       repeat=self.num_players-1))
 

    def set_state(self, p_state, hidden_state, p_id):
        self.ended = False
        self.curr_player = p_id
        self.cards = list(self.poss_hidden[hidden_state])
        player_card = (p_state % (self.num_players+1))+1
        self.cards.insert(p_id, player_card)

        p_pot = (p_state // (self.num_players+1))

        self.curr_bets = list(self.poss_pots[p_pot])
        self.curr_bets.insert(p_id, 1)
        
        self.betted = [bool(bet-1) for bet in self.curr_bets]
        self.checked = [False for betted in self.betted]
        for p in range(p_id):
            self.checked[p] = not self.betted[p]
        self.folded = [False for player in range(self.num_players)]
        if any(self.betted):
            first = self.betted.index(True)
            if p_id > first:
                for p in range(first, p_id):
                    self.folded[p] = not self.betted[p]
            else:
                for p in range(p_id):
                    self.folded[p] = not self.betted[p]
                for p in range(first, self.num_players):
                    self.folded[p] = not self.betted[p]

    def get_hidden(self, p_id):
        curr_cards = self.cards.copy()
        curr_cards.pop(p_id)
        hidden_state = self.poss_hidden.index(tuple(curr_cards))
        return hidden_state
