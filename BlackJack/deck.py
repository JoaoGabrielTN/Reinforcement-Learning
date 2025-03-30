import collections
from random import choice

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(v) for v in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()
    
    def __init__(self) -> None:
        self._cards = [(suit, rank) for suit in self.suits
                                   for rank in self.ranks]
    
    def __len__(self) -> int:
        return len(self._cards)
    
    def __getitem__(self, postion) -> Card:
        return self._cards[postion]
    
    def draw(self) -> Card:
        card = choice(self._cards)
        self._cards.remove(card)
        return card
    
    def refresh(self) -> None:
        del self._cards
        self._cards = [(suit, rank) for suit in self.suits
                                   for rank in self.ranks]