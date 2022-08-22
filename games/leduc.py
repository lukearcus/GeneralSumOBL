
class leduc(base):
    curr_player = 0
    curr_bets = [0,0]
    cards = [0,0]
    winner = 0
    ended = False
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
        self.curr_player = 0
        self.rewards = [0 for i in range(self.num_players)]
        self.ended = False
        
        self.folded = [False for i in range(self.num_players)]
        self.checked = [False for i in range(self.num_players)]
        self.betted = [False for i in range(self.num_players)]

    def observe(self):
        raise NotImplementedError

    def action(self, act):
        raise NotImplementedError

    def end_game(self):
        raise NotImplementedError

