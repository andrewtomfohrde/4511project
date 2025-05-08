from dictionarytrie import DictionaryTrie
from word import DictionaryTrie



global LETTER_VALUES
LETTER_VALUES = {"A": 1,
                 "B": 3,
                 "C": 3,
                 "D": 2,
                 "E": 1,
                 "F": 4,
                 "G": 2,
                 "H": 4,
                 "I": 1,
                 "J": 8,
                 "K": 5,
                 "L": 1,
                 "M": 3,
                 "N": 1,
                 "O": 1,
                 "P": 3,
                 "Q": 10,
                 "R": 1,
                 "S": 1,
                 "T": 1,
                 "U": 1,
                 "V": 4,
                 "W": 4,
                 "X": 8,
                 "Y": 4,
                 "Z": 10,
                 "#": 0}

class Word:
    """
    Class representing a word being played on the board.
    Handles validation and interaction with the board structure.
    """
    def __init__(self, word, location, player, direction, board):
        self.word = word.upper()
        self.location = location
        self.player = player
        self.direction = direction.lower()
        self.board = board
        self.blank_positions = [] # do we need
    
    def validate_move_connectivity(self):
        """
        Validates that a move properly connects to existing tiles on the board.
        
        Args:
            board: Game board
            word: The word being played
            position: Starting position (col, row)
            direction: "right" or "down"
            placed_tiles: List of tiles placed [(position, letter)]
            
        Returns:
            True if the move connects to existing tiles, False otherwise
        """
        # Skip connectivity check for first word (should be checked separately)
        first_word = True
        for row in range(15):
            for col in range(15):
                node = self.board.get_node(row, col)
                if node and node.tile:
                    first_word = False
                    break
            if not first_word:
                break
                
        if first_word:
            # For first word, check if it covers center star (7,7)
            row, col = self.location
            if self.direction == "right":
                return col <= 7 <= col + len(self.word) - 1 and row == 7
            else:  # down
                return row <= 7 <= row + len(self.word) - 1 and col == 7
                
        # For subsequent words, check connectivity
        connects_to_existing = False
        all_positions = []
        
        # Get all positions this word will occupy
        row, col = self.location
        if self.direction == "right":
            all_positions = [(row, col + i) for i in range(len(self.word))]
        else:  # down
            all_positions = [(row + i, col) for i in range(len(self.word))]
        
        # Check if any position in the word overlaps with existing tiles
        for pos in all_positions:
            r, c = pos
            node = self.board.get_node(r, c)
            if node and node.tile:
                connects_to_existing = True
                break
        
        # Check if any adjacent position has a tile
        if not connects_to_existing:
            for pos in all_positions:
                r, c = pos
                # Check adjacent cells (up, down, left, right)
                adjacent_positions = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                for adj_row, adj_col in adjacent_positions:
                    if 0 <= adj_row < 15 and 0 <= adj_col < 15:
                        node = self.board.get_node(adj_row, adj_col)
                        if node and node.tile:
                            connects_to_existing = True
                            break
                if connects_to_existing:
                    break
                    
        return connects_to_existing
    
    def check_word(self):
        """
        Enhanced check_word method that validates the primary word and all
        secondary words formed by the placement.
        """
        global round_number, players, dictionary
        dictionary = DictionaryTrie()
        
        dictionary = load_dictionary_from_file("build/scrabbledict.txt")
            
        # Handle out of bounds checks
        if self.location[0] > 14 or self.location[1] > 14 or self.location[0] < 0 or self.location[1] < 0 or \
        (self.direction == "down" and (self.location[0] + len(self.word) - 1) > 14) or \
        (self.direction == "right" and (self.location[1] + len(self.word) - 1) > 14):
            print("Location out of bounds")
            return False, None
        
        # Handle blank tiles similar to original code
        
        full = "" # full in-line word; checking validity of placement
        
        # Check if this is the first word
        connects = self.validate_move_connectivity()    
        
        if not connects:
            print("Word must connect to existing tiles on the board.")
            return False, None
                
        if self.direction == "right":
            row, col = self.location
            curr_tile = self.board.get_node(row, col)
            i = 1
            while (col - i >= 0) and (curr_tile.left.tile):
                curr_tile = curr_tile.left
                i += 1
                if (col - i == 0) or (not curr_tile.left.tile):
                    break
            i = 0
            while ((curr_tile and curr_tile.right and curr_tile.right.tile) or (curr_tile and curr_tile.position == (row, col + i))) and (col + i <= 14):
                if curr_tile.position == (row, col + i) and len(self.word) > i:
                    full += self.word[i]
                elif (curr_tile.right and curr_tile.right.tile and curr_tile.tile):
                    full += curr_tile.tile.letter
                    i = 0
                else:
                    break
                curr_tile = curr_tile.right
                i += 1
            
        else:
            row, col = self.location
            curr_tile = self.board.get_node(row, col)
            i = 1
            while (row - i >= 0) and (curr_tile.up.tile):
                curr_tile = curr_tile.up
                i += 1
                if (row - i == 0) or (not curr_tile.up.tile):
                    break
            i = 0
            while ((curr_tile and curr_tile.down and curr_tile.down.tile) or (curr_tile and curr_tile.position == (row + i, col))) and (row + i <= 14):
                if curr_tile.position == (row + i, col) and len(self.word) > i:
                    full += self.word[i]
                elif (curr_tile.down and curr_tile.down.tile and curr_tile.tile):
                    full += curr_tile.tile.get_letter()
                    i = 0
                else:
                    break
                curr_tile = curr_tile.down
                i += 1
        
        if (full != self.get_word()):
            print("Invalid word. Give a playable word.")
            return False, None

        # Check main word in dictionary
        if not dictionary.is_word(self.word.upper()):
            print("Word not found in dictionary.")
            return False, None
        
        # Get letters from player's rack that can be used
        available_letters = [tile.get_letter() for tile in self.player.get_rack_arr()]
        
        # Check if player has required tiles for the main word
        required_letters = list(self.word)
        place_tiles = []
        
        # Identify which letters will come from the board
        if self.direction == "right":
            for i, letter in enumerate(self.word):
                node = self.board.get_node(self.location[0], self.location[1] + i)
                if node.tile:
                    if node.tile.letter != self.word[i]:
                        print(f"Invalid move, played {self.word[i]} instead of {node.tile.letter}. Try again!")
                        return False, None
                    else:
                        required_letters[i] = None  # Mark as provided by board
                        place_tiles.append(((self.location[0], self.location[1] + i), None))
                else:
                    place_tiles.append(((self.location[0], self.location[1] + i), self.word[i]))
        else:  # down
            for i, letter in enumerate(self.word):
                node = self.board.get_node(self.location[0] + i, self.location[1])
                if node.tile:
                    if node.tile.letter != self.word[i]:
                        print(f"Invalid move, played {self.word[i]} instead of {node.tile.letter}. Try again!")
                        return False, None
                    else:
                        required_letters[i] = None  # Mark as provided by board
                        place_tiles.append(((self.location[0] + i, self.location[1]), None))
                else:
                    place_tiles.append(((self.location[0] + i, self.location[1]), self.word[i]))
        
        # Remove None values
        required_letters = [letter for letter in required_letters if letter is not None]
        
        # Check if rack has required letters
        for letter in required_letters:
            if letter in available_letters:
                available_letters.remove(letter)
            elif '#' in available_letters:  # Use a blank tile
                available_letters.remove('#')
            else:
                print(f"You don't have the required tiles to spell {self.word}.")
                return False, None
        
        # Now check and validate all secondary words formed
        if self.board.get_node(7,7).tile:
            secondary_words = self.find_secondary_words(place_tiles)
            if secondary_words:
                print(secondary_words)
            for word in secondary_words:
                if not dictionary.is_word(word) and len(word) > 1:
                    print(f"Secondary word '{word}' not found in dictionary. Try again.")
                    return False, place_tiles
        
        # Everything is valid
        return True, place_tiles
    
    def find_secondary_words(self, letters_to_place):
        """
        Find all secondary words created by the placement of the main word.
        Returns a list of tuples (word, positions) where positions is a list
        of (row, col) coordinates.
        """
        secondary_words = []
        curr_word = ""

        # board_copy.place_word(self.word, self.location, self.direction, self.player)
        # def place_word(self, word, location, direction, player):
        row, col = self.location
        
        for i, placed in enumerate(letters_to_place):
            if placed[1] is None:
                continue  # Skip tiles that are already on the board

            (row, col), letter = placed
            word = letter  # Start with the placed letter
            node = self.board.get_node(row, col)

            # If the main word is horizontal, look vertically
            if self.direction == "right":
                # Look upward
                r = row - 1
                while r >= 0 and self.board.get_node(r, col).tile:
                    word = self.board.get_node(r, col).tile.get_char() + word
                    r -= 1

                # Look downward
                r = row + 1
                while r < 15 and self.board.get_node(r, col).tile:
                    word += self.board.get_node(r, col).tile.get_char()
                    r += 1

            # If the main word is vertical, look horizontally
            elif self.direction == "down":
                # Look left
                c = col - 1
                while c >= 0 and self.board.get_node(row, c).tile:
                    word = self.board.get_node(row, c).tile.get_char() + word
                    c -= 1

                # Look right
                c = col + 1
                while c < 15 and self.board.get_node(row, c).tile:
                    word += self.board.get_node(row, c).tile.get_char()
                    c += 1

            # Only add if an actual secondary word was formed
            if len(word) > 1:
                secondary_words.append(word)

        return secondary_words

    def calculate_word_score(self, letters_to_place):
        """
        Calculate the score for the main word and all secondary words.
        """
        global LETTER_VALUES
        total_score = 0
        curr_score = 0
        sec_score = 0
        xls = 1
        fxws = 1
        sxws = 1
        sec = False
        sec_val = 0
        tiles_used = 0
        
        # Create a temporary board to place the word for scoring
        board = self.board
    
        for tile in letters_to_place:
            sec = False
            sec_score = 0
            xls = 1
            sxws = 1
            location, letter = tile
            row, col = location
            curr_node = board.get_node(row, col)
            if letter != None:
                if curr_node.score_multiplier in ["TWS", "DWS", "TLS", "DLS"] and curr_node.tile:
                    if curr_node.score_multiplier != "": 
                        if curr_node.score_multiplier == "TWS":
                            fxws *= 3
                            sxws *= 3
                        elif curr_node.score_multiplier == "DWS":
                            fxws *= 2
                            sxws *= 2
                        elif curr_node.score_multiplier == "TLS":
                            xls *= 3
                        else:
                            xls *= 2
                if self.direction == "right":
                    while curr_node.up and curr_node.up.tile:
                        curr_node = curr_node.up
                    while curr_node and curr_node.tile:
                        tile = curr_node.tile
                        letter_score = LETTER_VALUES[tile.get_letter()]
                        if curr_node.position == (row, col):
                            letter_score = letter_score * xls
                            sec_val = letter_score
                            curr_score += letter_score
                        else:
                            sec_score += (letter_score * sxws)
                            sec = True
                        curr_node = curr_node.down
                    if sec:
                        sec_score += sec_val
                    total_score += (sec_score * sxws)
                else:
                    while curr_node.left and curr_node.left.tile:
                        curr_node = curr_node.left
                    while curr_node and curr_node.tile:
                        tile = curr_node.tile
                        letter_score = LETTER_VALUES[tile.get_letter()]
                        if curr_node.position == (row, col):
                            sec_val = letter_score * xls
                            letter_score = letter_score * xls
                            curr_score += letter_score
                        else:
                            sec_score += (letter_score * sxws)
                            sec = True
                        curr_node = curr_node.right
                    if sec:
                        sec_score += sec_val
                    total_score += (sec_score * sxws)
                tiles_used += 1
            else:
                tile = curr_node.tile
                letter_score = LETTER_VALUES[tile.get_letter()]
                curr_score += letter_score
        total_score += (curr_score * fxws)
        
        if tiles_used == 7:
            total_score += 50
        
        return total_score

    def set_word(self, word):
        """Sets the word."""
        self.word = word.upper()

    def set_location(self, location):
        """Sets the starting location of the word."""
        self.location = location

    def set_direction(self, direction):
        """Sets the direction of the word (right or down)."""
        self.direction = direction.lower()

    def get_word(self):
        """Returns the word."""
        return self.word
    
    def get_blank_pos(self):
        return self.blank_positions
    
    def set_blank_pos(self, pos):
        self.blank_positions = pos

def load_dictionary_from_file(file_path):
    """
    Load dictionary from a file and create a trie data structure.
    Args:
        file_path: Path to the dictionary file
    Returns:
        A DictionaryTrie object containing all words from the file
    """
    dictionary = DictionaryTrie()
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip whitespace and ignore empty lines
                word = line.strip()
                if word:
                    dictionary.add_word(word)
        
        print(f"Successfully loaded dictionary from {file_path}")
        return dictionary
    
    except FileNotFoundError:
        print(f"Error: Dictionary file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading dictionary: {str(e)}")
        return None