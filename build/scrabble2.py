from random import shuffle
import re
import copy

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

class Tile:
    """
    Class that allows for the creation of a tile. Initializes using an uppercase string of one letter,
    and an integer representing that letter's score.
    """
    def __init__(self, letter, letter_values):
        #Initializes the tile class. Takes the letter as a string, and the dictionary of letter values as arguments.
        self.letter = letter.upper() # letter tile represents (USED FOR BLANKS)
        if self.letter in letter_values:
            self.score = letter_values[self.letter]
        else:
            self.score = 0
        self.char = letter.upper() #tile itself

    def get_char(self):
        #Returns what char the tile represents (ex. blank = A)
        return self.char

    def get_score(self):
        #Returns the tile's score value.
        return self.score

    def get_letter(self):
        #Returns the tile's actual letter
        return self.letter
        
class Bag:
    """
    Creates the bag of all tiles that will be available during the game. Contains 98 letters and two blank tiles.
    Takes no arguments to initialize.
    """
    def __init__(self):
        #Creates the bag full of game tiles, and calls the initialize_bag() method, which adds the default 100 tiles to the bag.
        #Takes no arguments.
        self.bag = []
        self.initialize_bag()

    def add_to_bag(self, tile, quantity):
        #Adds a certain quantity of a certain tile to the bag. Takes a tile and an integer quantity as arguments.
        for i in range(quantity):
            self.bag.append(tile)

    def initialize_bag(self):
        #Adds the intiial 100 tiles to the bag.
        global LETTER_VALUES
        self.add_to_bag(Tile("A", LETTER_VALUES), 9)
        self.add_to_bag(Tile("B", LETTER_VALUES), 2)
        self.add_to_bag(Tile("C", LETTER_VALUES), 2)
        self.add_to_bag(Tile("D", LETTER_VALUES), 4)
        self.add_to_bag(Tile("E", LETTER_VALUES), 12)
        self.add_to_bag(Tile("F", LETTER_VALUES), 2)
        self.add_to_bag(Tile("G", LETTER_VALUES), 3)
        self.add_to_bag(Tile("H", LETTER_VALUES), 2)
        self.add_to_bag(Tile("I", LETTER_VALUES), 9)
        self.add_to_bag(Tile("J", LETTER_VALUES), 1)
        self.add_to_bag(Tile("K", LETTER_VALUES), 1)
        self.add_to_bag(Tile("L", LETTER_VALUES), 4)
        self.add_to_bag(Tile("M", LETTER_VALUES), 2)
        self.add_to_bag(Tile("N", LETTER_VALUES), 6)
        self.add_to_bag(Tile("O", LETTER_VALUES), 8)
        self.add_to_bag(Tile("P", LETTER_VALUES), 2)
        self.add_to_bag(Tile("Q", LETTER_VALUES), 1)
        self.add_to_bag(Tile("R", LETTER_VALUES), 6)
        self.add_to_bag(Tile("S", LETTER_VALUES), 4)
        self.add_to_bag(Tile("T", LETTER_VALUES), 6)
        self.add_to_bag(Tile("U", LETTER_VALUES), 4)
        self.add_to_bag(Tile("V", LETTER_VALUES), 2)
        self.add_to_bag(Tile("W", LETTER_VALUES), 2)
        self.add_to_bag(Tile("X", LETTER_VALUES), 1)
        self.add_to_bag(Tile("Y", LETTER_VALUES), 2)
        self.add_to_bag(Tile("Z", LETTER_VALUES), 1)
        self.add_to_bag(Tile("#", LETTER_VALUES), 2)
        shuffle(self.bag)

    def take_from_bag(self):
        #Removes a tile from the bag and returns it to the user. This is used for replenishing the rack.
        return self.bag.pop()

    def get_remaining_tiles(self):
        #Returns the number of tiles left in the bag.
        return len(self.bag)
    
class Rack:
    """
    Creates each player's 'dock', or 'hand'. Allows players to add, remove and replenish the number of tiles in their hand.
    """
    def __init__(self, bag):
        #Initializes the player's rack/hand. Takes the bag from which the racks tiles will come as an argument.
        self.rack = []
        self.bag = bag
        self.initialize()

    def add_to_rack(self):
        #Takes a tile from the bag and adds it to the player's rack.
        self.rack.append(self.bag.take_from_bag())

    def initialize(self):
        #Adds the initial 7 tiles to the player's hand.
        for i in range(7):
            self.add_to_rack()

    def get_rack_str(self):
        #Displays the user's rack in string form.
        return ", ".join(str(item.get_letter()) for item in self.rack)

    def get_rack_arr(self):
        #Returns the rack as an array of tile instances
        return self.rack

    def remove_from_rack(self, tile):
        #Removes a tile from the rack (for example, when a tile is being played).
        self.rack.remove(tile)

    def get_rack_length(self):
        #Returns the number of tiles left in the rack.
        return len(self.rack)

    def replenish_rack(self):
        #Adds tiles to the rack after a turn such that the rack will have 7 tiles (assuming a proper number of tiles in the bag).
        while self.get_rack_length() < 7 and self.bag.get_remaining_tiles() > 0:
            self.add_to_rack()
            
class Player:
    """
    Creates an instance of a player. Initializes the player's rack, and allows you to set/get a player name.
    """
    def __init__(self, bag):
        #Intializes a player instance. Creates the player's rack by creating an instance of that class.
        #Takes the bag as an argument, in order to create the rack.
        self.name = ""
        self.rack = Rack(bag)
        self.score = 0

    def set_name(self, name):
        #Sets the player's name.
        self.name = name

    def get_name(self):
        #Gets the player's name.
        return self.name

    def get_rack_str(self):
        #Returns the player's rack.
        return self.rack.get_rack_str()

    def get_rack_arr(self):
        #Returns the player's rack in the form of an array.
        return self.rack.get_rack_arr()

    def increase_score(self, increase):
        #Increases the player's score by a certain amount. Takes the increase (int) as an argument and adds it to the score.
        self.score += increase

    def get_score(self):
        #Returns the player's score
        return self.score
    
class BoardNode(object):
    """
    A node in the linked list representation of the Scrabble board.
    Each node represents a single cell on the board.
    """
    def __init__(self):
        self.tile = None
        self.right = None
        self.down = None
        self.left = None
        self.up = None
        self.char = ' '
        self.occupied = False
        self.position = (0, 0) # (row, column)
        self.score_multiplier = ""
        # TRIPLE_WORD_SCORE = ((0,0), (7, 0), (14,0), (0, 7), (14, 7), (0, 14), (7, 14), (14,14))
        # DOUBLE_WORD_SCORE = ((1,1), (2,2), (3,3), (4,4), (1, 13), (2, 12), (3, 11), (4, 10), (7, 7), (13, 1), (12, 2), (11, 3), (10, 4), (13,13), (12, 12), (11,11), (10,10))
        # TRIPLE_LETTER_SCORE = ((1,5), (1, 9), (5,1), (5,5), (5,9), (5,13), (9,1), (9,5), (9,9), (9,13), (13, 5), (13,9))
        # DOUBLE_LETTER_SCORE = ((0, 3), (0,11), (2,6), (2,8), (3,0), (3,7), (3,14), (6,2), (6,6), (6,8), (6,12), (7,3), (7,11), (8,2), (8,6), (8,8), (8, 12), (11,0), (11,7), (11,14), (12,6), (12,8), (14, 3), (14, 11))
        
    
    def place_tile(self, tile):
        if self.occupied:
            print(f"Tile at {self.position} already occupied\n")
            return False
        self.tile = tile
        self.char = tile
        self.occupied = True
        
    def place_blank(self, tile, char):
        if self.occupied:
            print(f"Tile at {self.position} already occupied\n")
            return False
        self.tile = tile
        self.char = char
        self.occupied = True
        
    def get_display_str(self):
        """Return a string representation of this cell."""
        if self.occupied:
            return f"{self.char}/{self.tile}"
        elif self.position == (7, 7):  # Center square
            return " * "
        elif self.score_multiplier:
            return self.score_multiplier
        else:
            return "   "
            
class ScrabbleBoard:
    """
    A linked list implementation of a Scrabble board.
    The board is a 15x15 grid of cells, where each cell is a BoardNode.
    """
    def __init__(self):
        # Constants for premium squares
        self.TRIPLE_WORD_SCORE = ((0,0), (7, 0), (14,0), (0, 7), (14, 7), (0, 14), (7, 14), (14,14))
        self.DOUBLE_WORD_SCORE = ((1,1), (2,2), (3,3), (4,4), (1, 13), (2, 12), (3, 11), (4, 10), (7, 7), (13, 1), (12, 2), (11, 3), (10, 4), (13,13), (12, 12), (11,11), (10,10))
        self.TRIPLE_LETTER_SCORE = ((1,5), (1, 9), (5,1), (5,5), (5,9), (5,13), (9,1), (9,5), (9,9), (9,13), (13, 5), (13,9))
        self.DOUBLE_LETTER_SCORE = ((0, 3), (0,11), (2,6), (2,8), (3,0), (3,7), (3,14), (6,2), (6,6), (6,8), (6,12), (7,3), (7,11), (8,2), (8,6), (8,8), (8, 12), (11,0), (11,7), (11,14), (12,6), (12,8), (14, 3), (14, 11))
        
        self.size = 15
        self.init_board()
        self.add_premium_squares()
        
    def init_board(self):
        """Initialize the board as a linked list structure."""
        # First, create all the nodes
        nodes = [[BoardNode() for _ in range(self.size)] for _ in range(self.size)]
        
        # Set positions for each node
        for i in range(self.size):
            for j in range(self.size):
                nodes[i][j].position = (i, j)
        
        # Connect nodes horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                nodes[i][j].right = nodes[i][j + 1]
                nodes[i][j + 1].left = nodes[i][j]
        
        # Connect nodes vertically
        for i in range(self.size - 1):
            for j in range(self.size):
                nodes[i][j].down = nodes[i + 1][j]
                nodes[i + 1][j].up = nodes[i][j]
        
        # Store the top-left node as the starting point
        self.start_node = nodes[0][0]

    def get_board(self):
        """Get a string representation of the board."""
        # Header row with column numbers
        board_str = "   |  " + "  |  ".join(str(i) for i in range(10)) + "  | " + "  | ".join(str(i) for i in range(10, 15)) + " |"
        board_str += "\n   _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n"
        
        # Each row of the board
        for i in range(self.size):
            row_cells = []
            current = self.get_node(i, 0)
            
            # Collect string representations of each cell in the row
            for j in range(self.size):
                row_cells.append(current.get_display_str())
                current = current.right
            
            # Row prefix with row number
            if i < 10:
                row_str = str(i) + "  | " + " | ".join(row_cells) + " |"
            else:
                row_str = str(i) + " | " + " | ".join(row_cells) + " |"
            
            board_str += row_str
            
            # Add separator row if not the last row
            if i < self.size - 1:
                board_str += "\n   |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|\n"
        
        # Bottom border
        board_str += "\n   _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _"
        
        return board_str

    def add_premium_squares(self):
        """Add premium square designations to the board."""
        for row, col in self.TRIPLE_WORD_SCORE:
            node = self.get_node(row, col)
            node.score_multiplier = "TWS"
        
        for row, col in self.DOUBLE_WORD_SCORE:
            node = self.get_node(row, col)
            node.score_multiplier = "DWS"
        
        for row, col in self.TRIPLE_LETTER_SCORE:
            node = self.get_node(row, col)
            node.score_multiplier = "TLS"
        
        for row, col in self.DOUBLE_LETTER_SCORE:
            node = self.get_node(row, col)
            node.score_multiplier = "DLS"

    def place_word(self, word, location, direction, player, placed_tiles):
        """Places a word on the board and calculates its score."""
        direction = direction.lower()
        word = word.upper()
        used_tiles = []
        
        start_row, start_col = location
        
        # First check if the word contains blanks (#) and get their positions
        blankpos = [i for i, letter in enumerate(word) if letter == '#']
        
        # For each letter in the word
        for i, letter in enumerate(word):
            row, col = start_row, start_col
            if direction == "right":
                col += i
            elif direction == "down":  # direction == "down"
                row += i
            else:
                print("Give valid direction for word placement\n")
                return False
            
            # Get the node at this position
            node = self.get_node(row, col)
            
            # Find matching tile in player's rack
            is_blank = i in blankpos
            used_tile = None
            
            if node is not None and node.tile is not None:
                print("Tile already placed in previous move\n")
                
            else:
                for tile in player.rack.rack:
                    # For blank tiles, look for '#'
                    if is_blank and tile.get_letter() == '#':
                        used_tile = tile
                        node.place_blank(used_tile.get_letter(), placed_tiles[i][1])
                        print("Placing blank tile #\n")
                        break
                    # For regular tiles, look for matching letter
                    elif not is_blank and tile.get_letter() == letter:
                        used_tile = tile
                        node.place_tile(used_tile.get_letter())
                        print(f"Placing actual tile {used_tile.get_letter()}\n")
                        break
                
                if not used_tile:
                    # Try to use a blank tile if regular tile not found
                    if not is_blank:
                        for tile in player.rack.rack:
                            if tile.get_letter() == '#':
                                used_tile = tile
                                is_blank = True
                                print(f"Using {used_tile} to fill in gaps of your play.\n")
                                break
            
            if used_tile:
                used_tiles.append(used_tile)
                
                # Place the tile on the board
                #(used_tile, letter)        
        
        # # Apply word multiplier
        # total_score *= word_multiplier
        
        # Remove used tiles from rack
        for tile in used_tiles:
            print(f"Removing {tile.get_letter()} from rack\n")
            player.rack.remove_from_rack(tile)
        
        # Replenish rack
        player.rack.replenish_rack()

        return True

    def board_graph(self):
        #Returns the 2-dimensional board array.
        return self.board
    
    def get_node(self, row, col):
        """Get the node at the specified position."""
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return None
        
        # Navigate to the requested node
        current = self.start_node
        
        # Move down to the correct row
        for _ in range(row):
            current = current.down
        
        # Move right to the correct column
        for _ in range(col):
            current = current.right
        
        return current
    
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
        self.blank_positions = []
    
    def check_word(self):
        """
        Enhanced check_word method that validates the primary word and all
        secondary words formed by the placement.
        """
        global round_number, players, dictionary
        if "dictionary" not in globals():
            dictionary = open("build\\scrabbledict.txt").read().splitlines()

        # Handle out of bounds checks
        if self.location[0] > 14 or self.location[1] > 14 or self.location[0] < 0 or self.location[1] < 0 or \
        (self.direction == "down" and (self.location[0] + len(self.word) - 1) > 14) or \
        (self.direction == "right" and (self.location[1] + len(self.word) - 1) > 14):
            print("Location out of bounds.\n")
            return False, None
        
        # Handle blank tiles similar to original code
        
        full = "" # full in-line word; checking validity of placement
        
        # Check if this is the first word
        first_word = True
        
        # Check each position on the board instead of iterating through board rows
        for row in range(15):
            for col in range(15):
                node = self.board.get_node(row, col)
                if node.occupied:
                    first_word = False
                    break
            if not first_word:
                break
        
        # First word must be placed on the center star
        if first_word:
            if not ((self.direction == "right" and self.location[0] == 7 and 
                    self.location[1] <= 7 <= self.location[1] + len(self.word) - 1) or
                    (self.direction == "down" and self.location[1] == 7 and 
                    self.location[0] <= 7 <= self.location[0] + len(self.word) - 1)):
                print("First word must cover the center star.\n")
                return False, None
        
        # For subsequent words, check if the placement touches existing tiles
        else:
            connects_to_existing = False
            all_positions = []
            
            # Get all positions this word will occupy
            if self.direction == "right":
                all_positions = [(self.location[0], self.location[1] + i) for i in range(len(self.word))]
            else:  # down
                all_positions = [(self.location[0] + i, self.location[1]) for i in range(len(self.word))]
            
            # Check if any position in the word overlaps with existing tiles
            for pos in all_positions:
                row, col = pos
                node = self.board.get_node(row, col)
                if node.occupied:
                    connects_to_existing = True
                    break
            
            # Check if any adjacent position has a tile
            if not connects_to_existing:
                for pos in all_positions:
                    row, col = pos
                    # Check adjacent cells (up, down, left, right)
                    adjacent_positions = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                    for adj_row, adj_col in adjacent_positions:
                        if 0 <= adj_row < 15 and 0 <= adj_col < 15:
                            node = self.board.get_node(adj_row, adj_col)
                            if node.occupied:
                                connects_to_existing = True
                                break
                    if connects_to_existing:
                        break
            
            if not connects_to_existing:
                print("Word must connect to existing tiles on the board.\n")
                return False, None
                
        if self.direction == "right":
            row, col = self.location
            curr_tile = self.board.get_node(row, col)
            i = 1
            while (curr_tile.left.occupied) and (col - i >= 0):
                curr_tile = curr_tile.left
                i += 1
                if (not curr_tile.left.occupied) or (col - i == 0):
                    break
            i = 0
            while ((curr_tile.right.occupied) or (curr_tile.position == (row, col + i))) and (col + i <= 14):
                if curr_tile.position == (row, col + i) and len(self.word) > i:
                    full += self.word[i]
                elif (curr_tile.right.occupied):
                    full += curr_tile.char
                    i = 0
                else:
                    break
                curr_tile = curr_tile.right
                i += 1
            
        else:
            row, col = self.location
            curr_tile = self.board.get_node(row, col)
            i = 1
            while (curr_tile.up.occupied) and (row - i >= 0):
                curr_tile = curr_tile.up
                i += 1
                if (not curr_tile.up.occupied) or (row - i == 0):
                    break
            i = 0
            while ((curr_tile.down.occupied) or (curr_tile.position == (row + i, col))) and (row + i <= 14):
                if curr_tile.position == (row + i, col) and len(self.word) > i:
                    full += self.word[i]
                elif (curr_tile.down.occupied):
                    full += curr_tile.char
                    i = 0
                else:
                    break
                curr_tile = curr_tile.down
                i += 1
        
        if (full != self.get_word()):
            print("Invalid word. Give a playable word.\n")
            return False, None

        # Check main word in dictionary
        if self.word.upper() not in dictionary:
            print("Word not found in dictionary.\n")
            return False, None
        
        # Get letters from player's rack that can be used
        available_letters = [tile.get_letter() for tile in self.player.get_rack_arr()]
        
        # Check if player has required tiles for the main word
        required_letters = list(self.word)
        board_letters = []
        place_tiles = []
        
        # Identify which letters will come from the board
        if self.direction == "right":
            for i, letter in enumerate(self.word):
                node = self.board.get_node(self.location[0], self.location[1] + i)
                if node.occupied:
                    if node.char != self.word[i]:
                        print(f"Invalid move, played {self.word[i]} instead of {node.char}. Try again!")
                        return False, None
                    else:
                        board_letters.append(node.char)
                        required_letters[i] = None  # Mark as provided by board
                        place_tiles.append(((self.location[0], self.location[1] + i), None))
                else:
                    place_tiles.append(((self.location[0], self.location[1] + i), self.word[i]))
        else:  # down
            for i, letter in enumerate(self.word):
                node = self.board.get_node(self.location[0] + i, self.location[1])
                if node.occupied:
                    if node.char != self.word[i]:
                        print(f"Invalid move, played {self.word[i]} instead of {node.char}. Try again!")
                        return False, None
                    else:
                        board_letters.append(node.char)
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
                print(f"You don't have the required tiles to spell {self.word}.\n")
                return False, None
        
        # Now check and validate all secondary words formed
        if self.board.get_node(7,7).occupied:
            secondary_words = self.find_secondary_words(place_tiles)
            for word in secondary_words:
                if word not in dictionary and len(word) > 1:
                    print(f"Secondary word '{word}' not found in dictionary. Try again.\n")
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
                while r >= 0 and self.board.get_node(r, col).occupied:
                    word = self.board.get_node(r, col).char + word
                    r -= 1

                # Look downward
                r = row + 1
                while r < 15 and self.board.get_node(r, col).occupied:
                    word += self.board.get_node(r, col).char
                    r += 1

            # If the main word is vertical, look horizontally
            elif self.direction == "down":
                # Look left
                c = col - 1
                while c >= 0 and self.board.get_node(row, c).occupied:
                    word = self.board.get_node(row, c).char + word
                    c -= 1

                # Look right
                c = col + 1
                while c < 15 and self.board.get_node(row, c).occupied:
                    word += self.board.get_node(row, c).char
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
                if curr_node.score_multiplier in ["TWS", "DWS", "TLS", "DLS"] and curr_node.occupied:
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
                    while curr_node.up and curr_node.up.occupied:
                        curr_node = curr_node.up
                    while curr_node.occupied:
                        tile = curr_node.tile
                        letter_score = LETTER_VALUES[tile]
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
                    while curr_node.left and curr_node.left.occupied:
                        curr_node = curr_node.left
                    while curr_node.occupied:
                        tile = curr_node.tile
                        letter_score = LETTER_VALUES[tile]
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
            else:
                tile = curr_node.tile
                letter_score = LETTER_VALUES[tile]
                curr_score += letter_score
        total_score += (curr_score * fxws)
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
    
def turn(player, board, bag):
    """
    Begins a turn, by displaying the current board, getting the information to play a turn,
    and creates a recursive loop to allow the next person to play.
    """
    global round_number, players, skipped_turns, word
    placed = []
    
    # If the number of skipped turns is less than 6 in a row, and there are either tiles in the bag,
    # or no players have run out of tiles, play the turn.
    # Otherwise, end the game.
    if (skipped_turns < 6) or (player.rack.get_rack_length() == 0 and bag.get_remaining_tiles() == 0):
        
        # Displays whose turn it is, the current board, and the player's rack.
        print("\nRound " + str(round_number) + ": " + player.get_name() + "'s turn \n")
        print(board.get_board())  # Use the new method name
        
        # Gets information in order to play a word.
            # Create a Word object - but we need to adapt this for our new board implementation
            # Instead of board.board_array(), pass the board object directly
            
            # If the first word throws an error, creates a recursive loop until the information is given correctly.
        checked = False
        while checked == False:
            print("\n" + player.get_name() + "'s Letter Rack: " + player.get_rack_str())
            word_to_play = input("Word to play: ")
            
            if word_to_play:  # Only get location and direction if word is not empty
                location = []
                col = input("Column number: ")
                row = input("Row number: ")
                
                if (col == "" or row == "") or (col not in [str(x) for x in range(15)] or row not in [str(x) for x in range(15)]):
                    location = [-1, -1]
                else:
                    location = [int(row), int(col)]
                    
                direction = input("Direction of word (right or down): ")
                
                if word_to_play == "" and col == "" and row == "" and direction == "":
                    print("Your turn has been skipped.")
                    skipped_turns += 1
                
                word = Word(word_to_play, location, player, direction, board)
                
                blank_positions = [m.start() for m in re.finditer('#', word_to_play)]
                # full = "" # full in-line word; checking validity of placement / use in checkword
                blank_tiles_values = []
                
                
                if blank_positions:
                    print(f"{len(blank_positions)} BLANK(S) DETECTED")
                        
                    # Get values for each blank tile
                    for i, pos in enumerate(blank_positions):
                        blank_value = ""
                        blank_value = input(f"Please enter the letter value of blank tile {i+1}: ")
                        blank_value = blank_value.upper()
                        blank_tiles_values.append(blank_value)
                            
                    # Replace blanks with their values
                    global modified_word
                    modified_word = list(word_to_play)
                    for i, pos in enumerate(blank_positions):
                        # Adjust position if multiple blanks (earlier blanks shift positions)
                        modified_word[pos] = blank_tiles_values[i]
                        # Store the positions of blanks in the modified word
                    new_word = ''.join(modified_word)
                    word.set_word(new_word)

            checked, placed = word.check_word()
            
        # If the user has confirmed that they would like to skip their turn, skip it.
        # Otherwise, plays the correct word and prints the board.
        # Call the new place_word method with proper parameter order
        # The ScrabbleBoard.place_word expects (word, start_row, start_col, direction, player)
        success = board.place_word(word_to_play, location, direction, player, placed)
        word_score = word.calculate_word_score(placed)
        # word_score = word.calculate_word_score()
        if success:
            # The word score calculation is now handled inside place_word method
            # so we don't need word.calculate_word_score() anymore
            
            print(f"Word '{word.word}' placed for {word_score} points!")
            player.increase_score(word_score)
            
        else:
            print("Failed to place word. Please try again.")
            # Recurse with the same player to give them another chance
            turn(player, board, bag)
            return

        # Prints the current player's score
        print("\n" + player.get_name() + "'s score is: " + str(player.get_score()))
        
        # Gets the next player.
        if players.index(player) != (len(players) - 1):
            player = players[players.index(player) + 1]
        else:
            player = players[0]
            round_number += 1
            
        # Recursively calls the function in order to play the next turn.
        turn(player, board, bag)
        
    # If the number of skipped turns is over 6 or the bag has both run out of tiles and a player is out of tiles, end the game.
    else:
        end_game()

def start_game():
    #Begins the game and calls the turn function.
    global round_number, players, skipped_turns
    board = ScrabbleBoard()
    bag = Bag()

    #Asks the player for the number of players.
    num_of_players = 2

    #Welcomes players to the game and allows players to choose their name.
    print("\nWelcome to Scrabble! Please enter the names of the players below.")
    players = []
    for i in range(num_of_players):
        players.append(Player(bag))
        players[i].set_name(input("Please enter player " + str(i+1) + "'s name: "))

    #Sets the default value of global variables.
    round_number = 1
    skipped_turns = 0
    current_player = players[0]
    turn(current_player, board, bag)

def end_game():
    #Forces the game to end when the bag runs out of tiles.
    global players
    highest_score = 0
    winning_player = ""
    for player in players:
        if player.get_score > highest_score:
            highest_score = player.get_score()
            winning_player = player.get_name()
    print("The game is over! " + winning_player + ", you have won!")

    if input("\nWould you like to play again? (y/n)").upper() == "Y":
        start_game()

start_game()