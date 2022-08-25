from games.base import base
from itertools import product
import random

class leduc(base):
    curr_player = 0
    curr_bets = [0,0]
    cards = [0,0]
    pub_card = 0
    winner = 0
    ended = False
    pub_card_revealed = False
    num_players = 2
    num_cards = 6

    def __init__(self, _num_players=2):
        random.seed(1)
        self.num_players = _num_players
        self.num_states = [self.num_cards*(3+self.num_cards*3*3), self.num_cards*(3+self.num_cards*3*3)] 
        # 6 possible cards in hand, 3 information states for each player in first round,
        # then 3 possible starting pots for second round & new card in public (one of 5 left)
        self.num_actions = [3 for i in range(self.num_players)] # call/check, raise, fold (sometimes some not allowed)
    
    def start_game(self):
        self.curr_bets = [1 for i in range(self.num_players)]
        cards_available = list(range(self.num_cards)) # 6 cards, % 2 for suit, // 2 for number
        random.shuffle(cards_available)
        self.cards = cards_available[:self.num_players]
        self.pub_card = cards_available[self.num_players]
        self.end_first_bets = 0

        self.num_raises = 0
        self.curr_player = 0
        self.pub_card_revealed = False
        self.rewards = [0 for i in range(self.num_players)]
        self.ended = False
        
        self.folded = [False for i in range(self.num_players)]

    def observe(self):
        if not self.ended:
            if self.pub_card_revealed:
                return self.cards[self.curr_player], self.curr_bets, self.pub_card, 0
            else:
                return self.cards[self.curr_player], self.curr_bets, -1, 0
        else:
            return -1, None, None, self.rewards[self.curr_player]

    def action(self, act):
        if self.pub_card_revealed:
            raise_amount = 4
        else:
            raise_amount = 2

        if not self.ended:
            if act=="raise":
                if self.num_raises < 2:
                    self.num_raises += 1
                    self.curr_bets[self.curr_player] += raise_amount
                else:
                    act = "check"
            if act=="check":
                self.curr_bets[self.curr_player] = max(self.curr_bets)
                # stays at current amount if no raise, otherwise calls
            if act=="fold":
                self.folded[self.curr_player] = True
            
            if sum(self.folded) == self.num_players -1:
                self.end_game()
            else:
                if max(self.curr_bets) > 1 or self.curr_player == self.num_players-1:
                    non_folded_bets = [bet for p, bet in enumerate(self.curr_bets) if not self.folded[p]]
                    if len(set(non_folded_bets)) == 1:
                        if self.pub_card_revealed:
                            self.end_game()
                        else:
                            self.pub_card_revealed = True
                            self.end_first_bets = self.curr_bets[0]
        player_chosen = False
        while not player_chosen:
            self.curr_player += 1
            if self.curr_player == self.num_players:
                self.curr_player = 0
            if self.ended or not self.folded[self.curr_player]:
                player_chosen = True

    def end_game(self):

        non_folded_cards = [card for p, card in enumerate(self.cards) if not self.folded[p]]
        pairs = []
        for card in non_folded_cards:
            if card // 2 == self.pub_card // 2:
                pairs.append(card) # check for pairs with public card
        if len(pairs) > 0:
            best_card = max(pairs) # if there are pairs, then highest pair wins
        else:
            best_card = max(non_folded_cards) # otherwise, highest card wins
        self.winner = self.cards.index(best_card)
        self.ended = True
        losses = [-bet for bet in self.curr_bets]
        winnings = sum(self.curr_bets)
        self.rewards = losses
        self.rewards[self.winner] += winnings

class leduc_int(leduc):
    
    def __init__(self, _num_players=2):
        super().__init__(_num_players)
        self.first_poss_pots = list(product([1,3,5], repeat=self.num_players-1))
        self.second_poss_pots = list(product([0,4,8], repeat=self.num_players-1))
        #self.poss_pots = list(product([1,3,5,7,9,11,13],repeat=self.num_players-1))
    
    def observe(self):    
        card, game_pot, pub_card, reward = super().observe()
        if card != -1:
            pot = game_pot.copy()
            pot.pop(self.curr_player)
            if pub_card == -1:
                pot_ind = self.first_poss_pots.index(tuple(pot))
                return pot_ind*(self.num_cards)+card, reward
            else:
                end_round1_ind = (self.end_first_bets-1)/2
                true_second_poss_pots = [tuple([elem+self.end_first_bets for elem in pot]) \
                                         for pot in self.second_poss_pots]
                pot_ind = true_second_poss_pots.index(tuple(pot))
                poss_pub_cards = list(range(self.num_cards))
                poss_pub_cards.pop(card)
                try:
                    pub_card_ind = poss_pub_cards.index(pub_card) + 1
                except ValueError:
                    import pdb; pdb.set_trace()
                return int((pub_card_ind)*3*3*self.num_cards\
                        +end_round1_ind*3*self.num_cards+pot_ind*self.num_cards+card), reward
        else:
            return -1, reward

    def action(self, act):
        if act == 0:
            super().action("raise")
        elif act == 1:
            super().action("check")
        else:
            super().action("fold")

class leduc_fict(leduc_int):
    
    def __init__(self):
        super().__init__()
        self.poss_hidden = list(product(list(range(0,self.num_cards)), \
                                       repeat=self.num_players-1))
 

    def set_state(self, p_state, hidden_state, p_id):
        if self.num_players > 2:
            raise NotImplementedError
        else:
            self.ended = False
            self.curr_player = p_id
            self.cards = list(self.poss_hidden[hidden_state])
            player_card = (p_state % (self.num_cards))
            self.cards.insert(p_id, player_card)
            
            p_pot = (p_state // self.num_cards) % 3
            

            if p_state > self.num_cards*3:
                end_round1_ind = (p_state // (self.num_cards*3)) % 3
                end_round1_bets = (end_round1_ind*2)+1
                self.end_first_bets = end_round1_bets
                pub_card_ind = ((p_state // (self.num_cards*3*3)) % 5)-1
                poss_pub_cards = list(range(self.num_cards))
                poss_pub_cards.pop(player_card) 
                self.pub_card = poss_pub_cards[pub_card_ind]
                if self.pub_card in self.cards:
                    return -1 # impossible state
                self.pub_card_revealed = True
                bets = self.second_poss_pots[p_pot]
                bets = [bet+end_round1_bets for bet in bets]
            else:
                self.pub_card_revealed = False
                bets = self.first_poss_pots[p_pot]

                pub_card_sel = False
                while not pub_card_sel:
                    pub_card = random.randint(0, self.num_cards-1)
                    if pub_card not in self.cards:
                        pub_card_sel = True
                        self.pub_card = pub_card
                

            self.curr_bets = list(bets)
            if self.pub_card_revealed:
                self.curr_bets.insert(p_id, max(end_round1_bets, bets[0]-4))
            else:
                self.curr_bets.insert(p_id, max(1, bets[0]-2))
            
            self.folded = [False for folded in self.folded]
            return 0

    def get_hidden(self, p_id):
        curr_cards = self.cards.copy()
        curr_cards.pop(p_id)
        hidden_state = self.poss_hidden.index(tuple(curr_cards))
        return hidden_state
