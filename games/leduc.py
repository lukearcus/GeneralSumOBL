
class leduc(base):
    curr_player = 0
    curr_bets = [0,0]
    cards = [0,0]
    pub_card = 0
    winner = 0
    ended = False
    pub_card_revealed = False
    num_players = 2

    def __init__(self):
        random.seed(1)
        self.num_players = _num_players
        self.num_states = [6*(3+5*3*3), 6*(3+5*3*3)] 
        # 6 possible cards in hand, 3 information states for each player in first round,
        # then 3 possible starting pots for second round & new card in public (one of 5 left)
        self.num_actions = [3 for i in range(self.num_players)] # call/check, raise, fold (sometimes some not allowed)
    
    def start_game(self):
        self.curr_bets = [1 for i in range(self.num_players)]
        cards_available = list(range(6)) # 6 cards, % 2 for suit, // 2 for number
        random.shuffle(cards_available)
        self.cards = cards_available[:self.num_players]
        self.pub_card = cards_available[self.num_players]
        
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
                non_folded_bets = [bet for p, bet in enumerate(self.curr_bets) if not self.folded[p]]
                if len(set(non_folded_bets)) == 1:
                    if self.pub_card_revealed:
                        self.end_game()
                    else:
                        self.pub_card_revealed = True
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
    
    def __init__(self):
        super().__init__()
        self.first_poss_pots = list(product([1,3,5], repeat=self.num_players-1))
        self.second_poss_pots = list(product([
        self.poss_pots = list(product([1,3,5,7,9,11,13],repeat=self.num_players-1))
    
    def observe(self):    
        card, game_pot, pub_card, reward = super().observe()
        if card != -1:
            pot = game_pot.copy()
            pot.pop(self.curr_player)
            pot_ind = self.poss_pots.index(tuple(pot))
            return (pub_card+1)*9 +pot_ind*(6)+card, reward
        else:
            return -1, reward

    def action(self, act):
        if act == 0:
            super().action("raise")
        elif act == 1:
            super().action("check")
        else:
            super().action("fold")
