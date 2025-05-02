class DictionaryTrie:
    class Node:
        def __init__(self):
            self.children = {}
            self.is_terminal = False
            
        def get_child(self, letter):
            """Get the child node for a letter, or None if it doesn't exist."""
            return self.children.get(letter.upper())
    
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
    
    def is_word(self, word):
        """Check if a word is in the dictionary."""
        current = self.root
        for letter in word.upper():
            if letter not in current.children:
                return False
            current = current.children[letter]
        return current.is_terminal
