import numpy as np
import random

class player:
    state = None
    belief = None

    def __init__(self):
        pass

    def observe(self, observation):
        pass

    def action(self):
        sel_act = None
        return sel_act
    
    def reset(self):
        pass

    def wipe_mem(self):
        pass

class human(player):
    Q_mat = 0
    state_hist=[]
    r_hist = []
    def observe(self, observation):
        if observation[0] != -1:
            print("Current pot is: " + str(observation[1]))
            print("Your card is: " + str(observation[0]))
        else:
            print("Your payout is: " + str(observation[2]))
        self.state = observation
        self.state_hist.append(self.state)
        self.r_hist.append(observation[2])

    def action(self):
        print("Choose an action: ")
        if self.state[1][0] == self.state[1][1]:
            print("bet")
            print("check")
        else:
            print("bet")
            print("fold")
        sel_act = input()
        return sel_act
    
    def get_reward(self, reward):
        print("You got " + str(reward) + " coins")




class RL(player):
    opt_pol = None

    def __init__(self, learner, player_id):
        self.learner = learner
        self.opt_pol = learner.opt_pol
        self.id = player_id
        self.buffer = []

    def wipe_mem(self):
        self.buffer = []
        self.learner.wipe_memory()

    def reset(self):
        self.learner.reset()
        self.opt_pol = self.learner.opt_pol
    
    def observe(self, observation, fict=False):
        self.state = observation[0]
        if not fict:
            reward = observation[1]
            if self.buffer != []:
                self.buffer[-1]["s'"] = self.state
                self.buffer[-1]["r"] = reward
            if self.state != -1:
                self.buffer.append({"s": self.state, "a": -1, "r": 0, "s'": -1})
            else:
                self.learner.update_memory([(self.buffer, None)])
                self.opt_pol = self.learner.learn()
        self.r = observation[1]

    def action(self):
        probs = self.opt_pol[self.state, :]
        act = np.argmax(np.random.multinomial(1, pvals=probs))
        self.buffer[-1]["a"] = act
        return act

class fixed_pol(player):
    opt_pol = None

    def __init__(self, opt_pol):
        self.opt_pol = opt_pol

    def reset(self):
        pass

    def observe(self, observation):
        self.state = observation[0]
        self.r = observation[1]
    
    def action(self):
        probs = self.opt_pol[self.state, :]
        act = np.argmax(np.random.multinomial(1, pvals=probs))
        return act


class OBL(RL):
    belief = 0
    
    def __init__(self, learner, player_id, fict_game, belief_iters = 1000):
        self.belief_iters = belief_iters
        super().__init__(learner, player_id)
        self.fict_game = fict_game
    
    def set_other_players(self, other_players):
        self.other_players = other_players.copy()
        self.other_players.insert(self.id, "me")

    def observe(self, observation, fict=False):
        self.state = observation[0]
        self.r = observation[1]
        if not fict:
            if self.state != -1:
                belief_probs = self.belief[self.state, :]
                belief_state = np.argmax(np.random.multinomial(1, pvals=belief_probs))
                #Here we do OBL
                self.fict_game.set_state(self.state, belief_state, self.id)
                act = self.action()
                #if self.state == 0:
                #    import pdb; pdb.set_trace()
                
                self.fict_game.action(act)
                while self.fict_game.curr_player != self.id:
                    curr_player = self.other_players[self.fict_game.curr_player]
                    other_p_obs = self.fict_game.observe()
                    curr_player.observe(other_p_obs, fict=True)
                    
                    other_p_act = curr_player.action()
                    self.fict_game.action(other_p_act)
                next_obs = self.fict_game.observe()
                s_prime = next_obs[0]
                r = next_obs[1]
                self.buffer.append({"s":self.state, "a": act, "r":r, "s'":s_prime})
            else:
                self.learner.update_memory([(self.buffer, None)])
                self.opt_pol = self.learner.learn()
                
    def action(self):
        probs = self.opt_pol[self.state, :]
        act = np.argmax(np.random.multinomial(1, pvals=probs))
        return act

    def update_belief(self):
        num_hidden = len(self.fict_game.poss_hidden)
        num_states = self.opt_pol.shape[0]
        new_belief = np.ones((num_states, num_hidden))
        for i in range(self.belief_iters):
            self.fict_game.start_game()
            while not self.fict_game.ended:
                p_id = self.fict_game.curr_player
                if p_id == self.id:
                    player = self
                else:
                    player = self.other_players[p_id]
                player.observe(self.fict_game.observe(), fict=True)
                self.fict_game.action(player.action())
                if p_id == self.id:
                    hidden_state = self.fict_game.get_hidden(self.id)
                    new_belief[self.state, hidden_state] += 1
        new_belief /= np.sum(new_belief,1,keepdims=True)
        self.belief = new_belief
