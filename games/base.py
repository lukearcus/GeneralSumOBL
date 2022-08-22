
class base:

    def __init__(self):
        raise NotImplementedError

    def start_game(self):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    def action(self, act):
        raise NotImplementedError

    def end_game(self):
        raise NotImplementedError

