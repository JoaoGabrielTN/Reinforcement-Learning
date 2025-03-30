from deck import FrenchDeck
class CustomEnv:
    deck = FrenchDeck()

    def __init__(self):
        self.action_space = 3
        self.observation_space = 3
        self.player = []
        self.dealer = []

    def reset(self):
        self.deck.refresh()
        self.dealer.append(self.deck.draw())
        self.dealer.append(self.deck.draw())
        self.player.append(self.deck.draw())
        self.player.append(self.deck.draw())

    def step(self, action):
        if action == 0:
            self.player
        elif action == 1:
            print("stick")
        
    def is_done(self):
        """ If players sticks or goes bust. Then the episode ends."""
        pass 

    def render(self):
        pass

    def action(self):
        pass

    def evaluate(self):
        pass

    def observe(self):
        pass

    def count(self):
        pass