class DictionaryTrie:
    class Node:
        def __init__(self, points=0, dir="", pos=None, wrd=""):
            self.children = {}
            self.is_terminal = False
            self.score = points
            self.direction = dir
            self.position = pos
            self.word = wrd
            
        def get_child(self, letter):
            """Get the child node for a letter, or None if it doesn't exist."""
            return self.children.get(letter.upper())
        
        def set_attr(self, word, pos, dir, score):
            self.score = score
            self.word = word
            self.position = pos
            self.direction = dir
    
    def __init__(self, word_list=None):
        self.root = self.Node()
        if word_list:
            for word in word_list:
                self.add_word(word)
    
    def add_word(self, word):
        """Add a word to the dictionary."""
        current = self.root
        for letter in word.upper():
            if letter not in current.children:
                current.children[letter] = self.Node()
            current = current.children[letter]
        current.is_terminal = True
        return
    
    def is_word(self, word):
        """Check if a word is in the dictionary."""
        current = self.root
        for letter in word.upper():
            if letter not in current.children:
                return False
            current = current.children[letter]
        return current.is_terminal
    
    def get_node(self, prefix):
        """
        Get the node that corresponds to the end of the given prefix.
        Returns None if the prefix isn't in the trie.
        """
        current = self.root
        for letter in prefix.upper():
            current = current.get_child(letter)
            if current is None:
                return None
        return current
