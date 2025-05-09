from random import shuffle
import re
from word_generator import get_possible_words
from word_generator import load_dictionary
from mcts_scrabble import monte_carlo_tree_search




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
        self.char = tile.get_letter()
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
                        print(f"Placing tile # as {placed_tiles[i][1]}")
                        break
                    # For regular tiles, look for matching letter
                    elif not is_blank and tile.get_letter() == letter:
                        used_tile = tile
                        node.place_tile(used_tile.get_letter())
                        print(f"Placing tile {used_tile.get_letter()}")
                        break
                
                if not used_tile:
                    # Try to use a blank tile if regular tile not found
                    if not is_blank:
                        for tile in player.rack.rack:
                            if tile.get_letter() == '#':
                                used_tile = tile
                                is_blank = True
                                print(f"Using {used_tile} to fill in gaps of your play.")
                                break
            
            if used_tile:
                used_tiles.append(used_tile)
                
                # Place the tile on the board
                #(used_tile, letter)        
        
        # # Apply word multiplier
        # total_score *= word_multiplier
        
        # Remove used tiles from rack
        for tile in used_tiles:
            print(f"Removing {tile.get_letter()} from rack")
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
    
<<<<<<< HEAD:build/scrabble2.py
    def find_anchor_points(self):
        """Find all empty cells adjacent to placed tiles."""
        anchor_points = set()
        current_node = self.start_node
        is_empty_board = True
        
        # Traverse the board to find anchor points
        row_node = current_node
        while row_node:
            col_node = row_node
            while col_node:
                if col_node.occupied:
                    is_empty_board = False
                    break
                col_node = col_node.right
            if not is_empty_board:
                break
            row_node = row_node.down

        if is_empty_board:
            return [None, (7, 7)]
        
        row_node = current_node
        row_index = 0

        while row_node:
            col_node = row_node
            col_index = 0

            while col_node:
                if not col_node.occupied:
                    adjacent_cells = [
                        (col_node.right, "right"),
                        (col_node.down, "down"),
                        (col_node.left, "left"),
                        (col_node.up, "up"),
                    ]
                    for adjacent_node, direction in adjacent_cells:
                        if adjacent_node and adjacent_node.occupied:
                            anchor_points.add((row_index, col_index))
                            break
                col_node = col_node.right
                col_index += 1

            row_node = row_node.down
            row_index += 1

        print(anchor_points)
            
        return list(anchor_points)

=======
>>>>>>> e0ce265bbd84c2996bfe4641c0b962da87f8e7bc:build/oldscrabble.py
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
            dictionary = open("scrabbledict.txt").read().splitlines()

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
                tiles_used += 1
            else:
                tile = curr_node.tile
                letter_score = LETTER_VALUES[tile]
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
    
def turn(player, board, bag):
    """
    Begins a turn, by displaying the current board, getting the information to play a turn,
    and creates a recursive loop to allow the next person to play.
    """
    global round_number, players, skipped_turns, word

    if (skipped_turns < 6) or (player.rack.get_rack_length() == 0 and bag.get_remaining_tiles() == 0):
        print("\nRound " + str(round_number) + ": " + player.get_name() + "'s turn \n")
        print(board.get_board())
        print("\n" + player.get_name() + "'s Letter Rack: " + player.get_rack_str())

        # === AI PLAYER LOGIC ===
        if player.get_name().upper() == "AI":
            print("[AI is thinking...]")

            if "dictionary" not in globals():
                globals()['dictionary'] = load_dictionary("scrabbledict.txt")

            legal_moves = get_possible_words(board, player.rack, dictionary, player, Word)

            if not legal_moves:
                print("AI has no valid moves. Skipping turn.")
                skipped_turns += 1
            else:
                initial_state = {
                    'board': board,
                    'rack': player.rack,
                    'legal_moves': legal_moves,
                    'player': player
                }

                best_move = monte_carlo_tree_search(initial_state, iterations=500)

                word_to_play = best_move['word']
                location = list(best_move['position'])
                direction = best_move['direction']

                print(f"AI plays: {word_to_play} at {location} going {direction}")

                word = Word(word_to_play, location, player, direction, board)

                valid, placed = word.check_word()
                if valid:
                    board.place_word(word_to_play, location, direction, player, placed)
                    word_score = word.calculate_word_score(placed)
                    print(f"Word '{word.word}' placed for {word_score} points!")
                    player.increase_score(word_score)
                    skipped_turns = 0
                else:
                    print("AI attempted invalid word. Skipping.")
                    skipped_turns += 1

        # === HUMAN PLAYER LOGIC ===
        else:
            placed = []
            checked = False
            while not checked:
                print("\n" + player.get_name() + "'s Letter Rack: " + player.get_rack_str())
                word_to_play = input("Word to play: ")
                location = []
                col = input("Column number: ")
                row = input("Row number: ")
                if (col == "" or row == "") or (col not in [str(x) for x in range(15)] or row not in [str(x) for x in range(15)]):
                    location = [-1, -1]
                else:
                    location = [int(row), int(col)]
                direction = input("Direction of word (right or down): ")

                word = Word(word_to_play, location, player, direction, board)

                blank_positions = [m.start() for m in re.finditer('#', word_to_play)]
                blank_tiles_values = []
                if blank_positions:
                    print(f"{len(blank_positions)} BLANK(S) DETECTED")
                    for i, pos in enumerate(blank_positions):
                        blank_value = input(f"Please enter the letter value of blank tile {i+1}: ").upper()
                        blank_tiles_values.append(blank_value)
                    modified_word = list(word_to_play)
                    for i, pos in enumerate(blank_positions):
                        modified_word[pos] = blank_tiles_values[i]
                    new_word = ''.join(modified_word)
                    word.set_word(new_word)

                checked, placed = word.check_word()

            success = board.place_word(word_to_play, location, direction, player, placed)
            word_score = word.calculate_word_score(placed)
            if success:
                print(f"Word '{word.word}' placed for {word_score} points!")
                player.increase_score(word_score)
                skipped_turns = 0
            else:
                print("Failed to place word. Please try again.")
                turn(player, board, bag)
                return

        print("\n" + player.get_name() + "'s score is: " + str(player.get_score()))

        if players.index(player) != (len(players) - 1):
            player = players[players.index(player) + 1]
        else:
            player = players[0]
            round_number += 1

        turn(player, board, bag)
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

<<<<<<< HEAD:build/scrabble2.py
    def end_game(self):
        #Forces the game to end when the bag runs out of tiles.
        global LETTER_VALUES
        for player in self.players:
            curr_score = player.get_score()
            for tile in player.rack.rack:
                letter_score = LETTER_VALUES[tile]
                curr_score -= letter_score
            player.increase_score(-curr_score)

        highest_score = 0
        winning_player = ""
        for player in self.players:
            if player.get_score() > highest_score:
                highest_score = player.get_score()
                winning_player = player.get_name()
        print("The game is over! " + winning_player + ", you have won!")

        if input("\nWould you like to play again? (y/n)").upper() == "Y":
            self.start_game()
    
    def start_game_vs_ai(self):
        #Begins the game and calls the turn function.
        board = ScrabbleBoard()
        bag = Bag()
        self.players = []

        #Asks the player for the number of players.
        valid = False

        #Welcomes players to the game and allows players to choose their name.
        print("\nWelcome to Scrabble! Please enter the names of the players below.")

        human_player = Player(bag)
        human_player.set_name(input("Please enter your name: "))
        self.players.append(human_player)

        ai_player = Player(bag)
        while not valid:
            ai_to_play = input("Please enter AI to be player " + str(i+1) + ": ")
            if ai_to_play is "MCTS":
                valid = True

            elif ai_to_play is "Beam":
                valid = True

            elif ai_to_play is "A":
                valid = True

            elif ai_to_play is "GBFS":
                valid = True

            else:
                quit = input("Invalid input. To quit, enter 'q': ").strip().lower() == 'q'
                if quit:
                    return

        self.players.append(ai_player)

        #Sets the default value of global variables.
        current_player = self.players[0]
        self.turn(current_player, board, bag)

    def start_ai_game(self):
        #Begins the game and calls the turn function.
        board = ScrabbleBoard()
        bag = Bag()
        self.players = []

        #Asks the player for the number of players.
        valid = False

        #Welcomes players to the game and allows players to choose their name.
        print("\nWelcome to Scrabble! Please enter the AI of each player below. Options include 'MCTS', 'Beam', 'A', 'GBFS'")
        self.players = []
        for i in range(2):
            while not valid:
                ai_to_play = input("Please enter AI to be player " + str(i+1) + ": ")
                if ai_to_play is ["MCTS", "Beam", "A", "GBFS"]:
                    valid = True
                if ai_to_play is "MCTS":
                    valid = True
                    
                elif ai_to_play is "Beam":
                    valid = True

                elif ai_to_play is "A":
                    valid = True

                elif ai_to_play is "GBFS":
                    valid = True

                else:
                    quit = input("Invalid input. To quit, enter 'q': ").strip().lower() == 'q'
                    if quit:
                        return
            self.players.append(Player(bag))
            self.players[i].set_name(input("Please enter player " + str(i+1) + "'s name: "))

        #Sets the default value of global variables.
        current_player = self.players[0]
        self.turn(current_player, board, bag)


#######################


def main():
    scrabble = Game()

    ai_game = input("Would you like to play against an AI? (y/n): ").strip().lower() == 'y'

    ai_num = input("Do you want AI vs AI? (y/n): ").strip().lower() == 'y'

    if ai_game:
        ai_num = input("Do you want AI vs AI? (y/n): ").strip().lower() == 'y'
        if ai_num:
            scrabble.start_ai_game()

        else:
            scrabble.start_game_vs_ai()

    else: 
        scrabble.start_game()

if __name__ == "__main__":
    main()


########################

class DAWGNode:
    """Node in a Directed Acyclic Word Graph"""
    
    def __init__(self):
        self.children = {}  # Map from characters to child nodes
        self.is_terminal = False  # True if this node represents the end of a valid word
        self.count = 0  # Number of words this node is part of (useful for pruning)
    
    
    def __str__(self):
        return f"DAWGNode(word={self.is_word}, children={list(self.children.keys())})"

    def get_child(self, letter):
        """
        Get the child node for a given letter.
        Args:
            letter: The letter to follow
        Returns:
            The child DAWGNode if the letter exists, None otherwise
        """
        self.children.get(letter.lower())


class DAWG:
    """Directed Acyclic Word Graph implementation for efficient word storage and lookups"""
    
    def __init__(self):
        self.root = DAWGNode()
        self._minimize_cache = {}  # For minimization during construction
    
    def insert(self, word):
        """Insert a word into the DAWG"""
        if not word:
            return
        
        word = word.lower()  # Normalize to lowercase
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = DAWGNode()
            node = node.children[char]
            node.count += 1
        
        node.is_terminal = True # Mark as a valid word endpoint
    
    def contains(self, word):
        """Check if a word exists in the DAWG"""
        if not word:
            return False
        
        word = word.lower()
        node = self.root
        
        for char in word:
            child_node = node.get_child(char)
            if child_node is None:
                return False
            node = child_node
        
        return node.is_terminal
    
    def get_words_with_prefix(self, prefix):
        """Get all words with the given prefix"""
        prefix = prefix.lower()
        words = []
        node = self.root
        
        # Navigate to the node representing the prefix
        for char in prefix:
            child_node = node.get_child(char)
            if child_node is None:
                return words  # Prefix not found
            node = child_node
        
        # DFS to find all words from this node
        self._dfs_collect_words(node, prefix, words)
        return words

    def _dfs_collect_words(self, node, prefix, words):
        """Helper method to collect all words with DFS"""
        if node.is_terminal:
            words.append(prefix)
        
        for char, child_node in node.children.items():
            self._dfs_collect_words(child_node, prefix + char, words)

class BeamSearchScrabble:
    def __init__(self, rack, board, beam_width=10, max_depth=7):
        """
        Initialize the beam search algorithm.
        
        Parameters:
        - game: Your Scrabble game instance
        - beam_width: Number of candidates to keep at each step
        - max_depth: Maximum number of tiles to place in a single move
        """
        self.rack = rack
        self.board = board
        self.beam_width = beam_width
        self.max_depth = max_depth
    
    def find_best_move(self):
        """
        Find the best move given the current rack and board state.
        
        Parameters:
        - rack: List of tiles in the player's rack
        
        Returns:
        - best_move: The highest scoring valid move found
        """
        initial_candidates = [{'placed_tiles': [], 'score': 0, 'rack': self.rack.copy()}]
        best_move = None
        best_score = 0
        
        # Identify all anchor points (empty cells adjacent to existing tiles)
        anchor_points = self.board.find_anchor_points()
        
        # Try starting a word from each anchor point
        for anchor in anchor_points:
            candidates = initial_candidates.copy()
            
            for depth in range(self.max_depth):
                new_candidates = []
                
                for candidate in candidates:
                    # Generate next possible moves from this candidate
                    next_moves = self._generate_next_moves(candidate, anchor)
                    new_candidates.extend(next_moves)
                
                # Keep only the beam_width best candidates
                candidates = sorted(new_candidates, key=lambda x: x['score'], reverse=True)[:self.beam_width]
                
                # Update best move if we found a better one
                for candidate in candidates:
                    if candidate['score'] > best_score and self._is_valid_move(candidate):
                        best_move = candidate
                        best_score = candidate['score']
            
        return best_move
    
    def generate_moves(self, candidate, anchor):
        """
        Generate all possible next moves from the current candidate.
        
        Parameters:
        - candidate: Current candidate move
        - anchor: Position to start the word
        
        Returns:
        - List of new candidate moves
        """
        new_candidates = []
        remaining_rack = candidate['rack']
        
        # Get node at anchor position
        node = self._get_node_at_position(anchor)
        
        # Try each direction (right, down)
        for direction in ['right', 'down']:
            current_node = node
            
            # Skip if we can't go in this direction
            if not getattr(current_node, direction):
                continue
                
            # Try each tile in the rack
            for i, tile in enumerate(remaining_rack):
                # Skip if cell is already occupied
                if current_node.occupied:
                    continue
                    
                # Create a new candidate with this tile placed
                new_rack = remaining_rack.copy()
                new_rack.pop(i)
                
                new_placed = candidate['placed_tiles'].copy()
                new_placed.append((current_node.position, tile))
                
                # Calculate the score for this placement
                new_score = self._calculate_placement_score(new_placed)
                
                new_candidates.append({
                    'placed_tiles': new_placed,
                    'score': new_score,
                    'rack': new_rack
                })
                
            # Move to the next node in the direction
            current_node = getattr(current_node, direction)
        
        return new_candidates
    
    def left_part(self, partial, dawgnode, limit, anchor):
        self.extend_right(partial, dawgnode, limit, anchor)
        if limit > 0:
            for letter in set(self.rack):
                next_node = dawgnode.get_child()
        # results = [partial] if dawgnode.is_terminal else []
    
        # # Base case: we've reached our limit for the left part
        # if limit <= 0:
        #     return results
        
        # # Try extending with each letter in the rack
        # for letter in set(self.rack):  # Using set to avoid duplicates
        #     next_node = dawgnode.get_child(letter)
        #     if next_node:  # If this is a valid prefix in our dictionary
        #         # Create a new rack without the used letter
        #         new_rack = list(self.rack)
        #         new_rack.remove(letter)
                
        #         # Recursively find all extensions of this prefix
        #         extensions = generate_left_parts(
        #             next_node, 
        #             new_rack,
        #             limit - 1, 
        #             partial + letter
        #         )
        #         results.extend(extensions)
        # return results
    
    def _get_node_at_position(self, position):
        """Get the node at the specified board position."""
        current = self.game.start_node
        row, col = position
        
        # Move down to the correct row
        for _ in range(row):
            if current.down:
                current = current.down
        
        # Move right to the correct column
        for _ in range(col):
            if current.right:
                current = current.right
                
        return current
    
    def _calculate_placement_score(self, placed_tiles):
        """
        Calculate the score for a set of placed tiles.
        This should leverage your existing calculate_word_score function.
        """
        # This is a placeholder - you'll need to implement this based on your game's scoring logic
        # You'll want to identify all words formed by these placements and sum their scores
        
        # For each placed tile, check if it forms words horizontally and vertically
        total_score = 0
        words_formed = self._identify_words_formed(placed_tiles)
        
        for word in words_formed:
            if self.check_word(word):
                word_score = self.calculate_word_score(word)
                total_score += word_score
                
        return total_score
    
    def _identify_words_formed(self, placed_tiles):
        """
        Identify all words formed by the placed tiles.
        Returns a list of words (as strings).
        """
        # This is a placeholder - you'll need to implement this based on your board representation
        # For each placed tile, extend left/right and up/down to find complete words
        words = []
        
        # Implementation would depend on your specific board representation
        # and how you track words
        
        return words
    
    def _is_valid_move(self, candidate):
        """
        Check if a candidate move is valid according to Scrabble rules.
        """
        # A move is valid if:
        # 1. All formed words are valid
        # 2. All placed tiles are in a single row or column
        # 3. All placed tiles are connected to existing tiles (except first move)
        # 4. No tiles overlap with existing tiles
        
        # For simplicity, let's assume we're checking word validity in _calculate_placement_score
        # and we're handling overlap prevention in _generate_next_moves
        
        placed_positions = [pos for pos, _ in candidate['placed_tiles']]
        
        # Check if tiles are placed in a straight line
        if not self._is_straight_line(placed_positions):
            return False
            
        # Other validity checks would go here
        
        return True
    
    def _is_straight_line(self, positions):
        """Check if all positions are in a straight line (same row or same column)."""
        if not positions:
            return True
            
        rows = [pos[0] for pos in positions]
        cols = [pos[1] for pos in positions]
        
        return len(set(rows)) == 1 or len(set(cols)) == 1
=======
    #Sets the default value of global variables.
    round_number = 1
    skipped_turns = 0
    current_player = players[0]
    turn(current_player, board, bag)

def end_game():
    #Forces the game to end when the bag runs out of tiles.
    global players
    global LETTER_VALUES
    for player in players:
        curr_score = player.get_score()
        for tile in player.rack.rack:
            letter_score = LETTER_VALUES[tile]
            curr_score -= letter_score
        player.increase_score(-curr_score)

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
>>>>>>> e0ce265bbd84c2996bfe4641c0b962da87f8e7bc:build/oldscrabble.py
