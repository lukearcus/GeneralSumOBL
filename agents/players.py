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
        if isinstance(self.learner,list):
            for learner in self.learner:
                learner.wipe_memory()
        else:
            self.learner.wipe_memory()

    def reset(self):
        if isinstance(self.learner,list):
            for learner in self.learner:
                learner.reset()
        else:
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

    def observe(self, observation, fict=False):
        self.state = observation[0]
        self.r = observation[1]
    
    def action(self):
        probs = self.opt_pol[self.state, :]
        act = np.argmax(np.random.multinomial(1, pvals=probs))
        return act


class OBL(RL):
    belief = 0
    
    def __init__(self, learner, player_id, fict_game, belief_iters = 10000):
        self.belief_iters = belief_iters
        self.belief_buff = []
        self.pol_buff = []
        super().__init__(learner, player_id)
        self.fict_game = fict_game
        self.avg_pol = np.ones_like(self.opt_pol)/self.opt_pol.shape[1]    

    def set_other_players(self, other_players):
        self.other_players = other_players.copy()
        self.other_players.insert(self.id, "me")

    def observe(self, observation, fict=False):
        self.state = observation[0]
        self.r = observation[1]
        if not fict:
            if self.state != -1:
                belief_probs = self.belief[self.state, :]
                #Here we do OBL
                res = -1
                while res != 0:
                    belief_state = np.argmax(np.random.multinomial(1, pvals=belief_probs))
                    res = self.fict_game.set_state(self.state, belief_state, self.id)
                    if res == -1:
                        false_prob = belief_probs[belief_state]
                        belief_probs[:] += false_prob/(belief_probs.size-1)
                        belief_probs[belief_state] = 0 # set prob to 0 if it was an impossible state
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
    
    def add_to_mem(self):
        for i in range(self.belief_iters):
            self.fict_game.start_game()
            while not self.fict_game.ended:
                p_id = self.fict_game.curr_player
                if p_id == self.id:
                    player = self
                else:
                    player = self.other_players[p_id]
                player.observe(self.fict_game.observe(), fict=True)
                act = player.action()
                self.fict_game.action(act)
                if p_id == self.id:
                    hidden_state = self.fict_game.get_hidden(self.id)
                    self.belief_buff.append({'s':self.state, 'hidden':hidden_state})
                    self.pol_buff.append({'s':self.state, 'a':act, 'probs':self.opt_pol[self.state, :]}) 

    def update_belief(self):
        num_hidden = len(self.fict_game.poss_hidden)
        num_states = self.opt_pol.shape[0]
        new_belief = np.ones((num_states, num_hidden))
        for elem in self.belief_buff:
            new_belief[elem['s'], elem['hidden']] += 1
        new_belief /= np.sum(new_belief,1,keepdims=True)
        self.belief = new_belief
    
    def learn_avg_pol(self):
        N = np.zeros(self.opt_pol.shape)
        for elem in self.pol_buff:
            state = elem['s']
            N[state,:] += elem['probs']
        pi = N/np.sum(N,axis=1)[:,np.newaxis]
        for i, s in enumerate(pi):
            if np.all(np.isnan(s)):
                pi[i] = np.ones(s.shape)/s.size
        self.avg_pol = pi

    def update_mem_and_bel(self):
        self.add_to_mem()
        self.update_belief()
        self.learn_avg_pol()

class OT_RL(RL):
    belief = 0
    
    def __init__(self, learner, player_id, fict_game, belief_iters = 10000, averaging="inf_state"):
        self.curr_lvl = 0
        self.belief_iters = belief_iters
        self.belief_buff = []
        self.pol_buff = []
        self.learner = learner
        self.opt_pol = learner[0].opt_pol
        self.pols = [np.copy(learner[0].opt_pol)]
        self.avg_pols = []
        self.id = player_id
        self.buffer = [[]]
        self.fict_game = fict_game
        self.beliefs = []
        self.avg_pol = np.ones_like(self.opt_pol)/self.opt_pol.shape[1]    
        self.learn_avg = averaging == "FSP_style"

    def set_other_players(self, other_players):
        self.other_players = other_players.copy()
        self.other_players.insert(self.id, "me")

    def observe(self, observation, fict=False):
        self.state = observation[0]
        self.r = observation[1]
        if not fict:
            if self.state != -1:
                for lvl in range(self.curr_lvl):
                    belief_probs = self.beliefs[lvl][self.state, :]
                    #Here we do OBL
                    res = -1
                    while res != 0:
                        belief_state = np.argmax(np.random.multinomial(1, pvals=belief_probs))
                        res = self.fict_game.set_state(self.state, belief_state, self.id)
                        #if res == -1:
                            #false_prob = belief_probs[belief_state]
                            #belief_probs[:] += false_prob/(belief_probs.size-1)
                            #belief_probs[belief_state] = 0 # set prob to 0 if it was an impossible state
                    act = self.action()
                    #if self.state == 0:
                    #    import pdb; pdb.set_trace()
                    
                    self.fict_game.action(act)
                    while self.fict_game.curr_player != self.id:
                        curr_player = self.other_players[self.fict_game.curr_player]
                        other_p_obs = self.fict_game.observe()
                        curr_player.observe(other_p_obs, fict=True)
                        
                        other_p_act = curr_player.action(lvl)
                        self.fict_game.action(other_p_act)
                    next_obs = self.fict_game.observe()
                    s_prime = next_obs[0]
                    r = next_obs[1]
                    self.buffer[lvl].append({"s":self.state, "a": act, "r":r, "s'":s_prime})
            else:
                for lvl in range(self.curr_lvl):
                    self.learner[lvl].update_memory([(self.buffer[lvl], None)])
                    self.pols[lvl+1] = self.learner[lvl].learn()
                self.opt_pol = self.pols[self.curr_lvl]
                
    def action(self, lvl=-1):
        if lvl == -1:
            probs = self.opt_pol[self.state, :]
        else:
            if self.learn_avg:
                probs = self.avg_pols[lvl][self.state, :]
            else:
                avg_pol = sum(self.pols[:lvl+1])/(lvl+1)
                probs = avg_pol[self.state,:]
        act = np.argmax(np.random.multinomial(1, pvals=probs))
        return act
    
    def add_to_mem(self):
        for i in range(self.belief_iters):
            self.fict_game.start_game()
            while not self.fict_game.ended:
                p_id = self.fict_game.curr_player
                if p_id == self.id:
                    player = self
                else:
                    player = self.other_players[p_id]
                player.observe(self.fict_game.observe(), fict=True)
                act = player.action()
                self.fict_game.action(act)
                if p_id == self.id:
                    hidden_state = self.fict_game.get_hidden(self.id)
                    self.belief_buff.append({'s':self.state, 'hidden':hidden_state})
                    self.pol_buff.append({'s':self.state, 'probs':self.opt_pol[self.state, :]}) 

    def update_belief(self):
        num_hidden = len(self.fict_game.poss_hidden)
        num_states = self.opt_pol.shape[0]
        new_belief = np.ones((num_states, num_hidden))
        for elem in self.belief_buff:
            new_belief[elem['s'], elem['hidden']] += 1
        new_belief /= np.sum(new_belief,1,keepdims=True)
        self.belief = new_belief
    
    def learn_avg_pol(self):
        N = np.zeros(self.opt_pol.shape)
        for elem in self.pol_buff:
            state = elem['s']
            N[state,:] += elem['probs']
        pi = N/np.sum(N,axis=1)[:,np.newaxis]
        for i, s in enumerate(pi):
            if np.all(np.isnan(s)):
                pi[i] = np.ones(s.shape)/s.size
        self.avg_pol = pi

    def update_mem_and_bel(self):
        self.pols.append(self.opt_pol)
        self.curr_lvl += 1
        self.add_to_mem()
        self.update_belief()
        if self.learn_avg:
            self.learn_avg_pol()
            self.avg_pols.append(np.copy(self.avg_pol))
        else:
            self.avg_pols.append(0)
            for lvl in range(self.curr_lvl):
                avg_pol = sum(self.pols[:lvl+1])/(lvl+1)
                self.avg_pols[lvl] = avg_pol
        self.beliefs.append(np.copy(self.belief))
        if self.curr_lvl < len(self.learner):
            self.opt_pol = self.learner[self.curr_lvl].opt_pol

    
    def wipe_mem(self):
        super().wipe_mem()
        self.buffer = [[] for i in range(self.curr_lvl+1)]
