from random import shuffle
import re
import copy
from word_generator import load_dictionary, get_possible_words
from dictionarytrie import DictionaryTrie
from word import Word, load_dictionary_from_file
from algos import ScrabbleAI, MCTS, BeamSearchScrabble
from player import Tile, Bag, Rack, Player

# from mcts_scrabble import monte_carlo_tree_search
# mcts_scrabble.py
# from scrabble import Word


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

# def load_dictionary_from_file(file_path):
#     """
#     Load dictionary from a file and create a trie data structure.
#     Args:
#         file_path: Path to the dictionary file
#     Returns:
#         A DictionaryTrie object containing all words from the file
#     """
#     dictionary = DictionaryTrie()
    
#     try:
#         with open(file_path, 'r') as file:
#             for line in file:
#                 # Strip whitespace and ignore empty lines
#                 word = line.strip()
#                 if word:
#                     dictionary.add_word(word)
        
#         print(f"Successfully loaded dictionary from {file_path}")
#         return dictionary
    
#     except FileNotFoundError:
#         print(f"Error: Dictionary file not found at {file_path}")
#         return None
#     except Exception as e:
#         print(f"Error loading dictionary: {str(e)}")
#         return None

class ScrabbleBoard:
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
        nodes = [[self.BoardNode() for _ in range(self.size)] for _ in range(self.size)]
        
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
                                print(f"Using {used_tile.get_letter()} to fill in gaps of your play.")
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
    
    def find_anchor_points(self):

        """Find all empty cells adjacent to placed tiles."""
        anchor_points = set()
        current_node = self.start_node
        
        # Traverse the board to find anchor points
        for i in range(self.size):
            row_node = current_node
            for j in range(self.size):
                if not row_node.occupied:
                    # Check if any adjacent node is occupied
                    if ((row_node.right and row_node.right.occupied) or
                        (row_node.left and row_node.left.occupied) or
                        (row_node.up and row_node.up.occupied) or
                        (row_node.down and row_node.down.occupied)):
                        anchor_points.add(row_node.position)
                row_node = row_node.right
            current_node = current_node.down
        return list(anchor_points)


##############################################################################3###


class Game:
    def __init__(self, dict):
        self.round_number = 1
        self.players = []
        self.skipped_turns = 0
        self.dictionary = dict

    def turn(self, player, board, bag):
        """
        Begins a turn, by displaying the current board, getting the information to play a turn,
        and creates a recursive loop to allow the next person to play.
        """
        if (self.skipped_turns < 6) or (player.rack.get_rack_length() == 0 and bag.get_remaining_tiles() == 0):
            print("\nRound " + str(self.round_number) + ": " + player.get_name() + "'s turn \n")
            print(board.get_board())
            print("\n" + player.get_name() + "'s Letter Rack: " + player.get_rack_str())

            # === AI PLAYER LOGIC ===
            if player.get_name().upper() in ["AI", "AI_MCTS", "AI_Beam", "AI_GBFS", "AI_AStar"] :
                print("[AI is thinking...]")

                if self.dictionary is None:
                    self.dictionary = load_dictionary("scrabbledict.txt")

                legal_moves = player.find_all_moves()

                if not legal_moves:
                    print("AI has no valid moves. Skipping turn.")
                    self.skipped_turns += 1
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
                        self.skipped_turns = 0
                    else:
                        print("AI attempted invalid word. Skipping.")
                        self.skipped_turns += 1

            # === HUMAN PLAYER LOGIC ===
            else:
                placed = []
                checked = False
                while not checked:
                    # print("\n" + player.get_name() + "'s Letter Rack: " + player.get_rack_str())
                    word_to_play = input("Word to play: ")
                    
                    # Check if player wants to skip turn
                    if word_to_play.lower() == "skip":
                        print(f"{player.get_name()} skips their turn.")
                        self.skipped_turns += 1
                        checked = True
                        continue
                        
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
                
                if checked and not word_to_play.lower() == "skip":
                    success = board.place_word(word_to_play, location, direction, player, placed)
                    word_score = word.calculate_word_score(placed)
                    if success:
                        print(f"Word '{word.word}' placed for {word_score} points!")
                        player.increase_score(word_score)
                        self.skipped_turns = 0
                    else:
                        print("Failed to place word. Skipping turn.")
                        self.skipped_turns += 1

            print("\n" + player.get_name() + "'s score is: " + str(player.get_score()))

            if self.players.index(player) != (len(self.players) - 1):
                next_player = self.players[self.players.index(player) + 1]
            else:
                next_player = self.players[0]
                self.round_number += 1

            # Refill player's rack
            player.rack.replenish_rack()
            
            # Recursive call for next player's turn
            self.turn(next_player, board, bag)
        else:
            self.end_game()

    def start_game(self):
        """Begins a standard human vs human game and calls the turn function."""
        board = ScrabbleBoard()
        bag = Bag()

        # Asks the player for the number of players.
        num_of_players = 2

        # Welcomes players to the game and allows players to choose their name.
        print("\nWelcome to Scrabble! Please enter the names of the players below.")
        self.players = []
        for i in range(num_of_players):
            self.players.append(Player(bag))
            self.players[i].set_name(input("Please enter player " + str(i+1) + "'s name: "))

        # Sets the default values
        self.round_number = 1
        self.skipped_turns = 0
        current_player = self.players[0]
        self.turn(current_player, board, bag)

    def start_game_vs_ai(self):
        """Begins a game between human and AI and calls the turn function."""
        board = ScrabbleBoard()
        bag = Bag()
        self.players = []

        # Welcomes players to the game
        print("\nWelcome to Scrabble! Let's play against the AI.")

        # Create human player
        human_player = Player(bag)
        human_player.set_name(input("Please enter your name: "))
        self.players.append(human_player)

        # Create AI player
        ai_type = input(f"Select AI type for player (MCTS/Beam/AStar/GBFS): ")
        ai_player = ScrabbleAI("", dictionary, board, bag, ai_type)
        ai_player.set_name(f"AI_{ai_type}")
        self.players.append(ai_player)

        # Sets the default values
        self.round_number = 1
        self.skipped_turns = 0
        current_player = self.players[0]
        self.turn(current_player, board, bag)

    def start_ai_game(self):
        """Begins a game between two AI players."""
        board = ScrabbleBoard()
        bag = Bag()
        self.players = []

        # Welcomes to the AI vs AI game
        print("\nWelcome to Scrabble AI vs AI match!")
        
        # Create two AI players
        for i in range(2):
            valid = False
            while not valid:
                ai_type = input(f"Select AI type for player {i+1} (MCTS/Beam/AStar/GBFS): ")
                if ai_type in ["MCTS", "Beam", "AStar", "GBFS"]:
                    valid = True
                else:
                    quit = input("Invalid input. To quit, enter 'q': ").strip().lower() == 'q'
                    if quit:
                        return

            ai_player = ScrabbleAI(dictionary, board, bag, ai_type)
            ai_player.set_name(f"AI_{ai_type}")
            self.players.append(ai_player)

        # Sets the default values
        self.round_number = 1
        self.skipped_turns = 0
        current_player = self.players[0]
        self.turn(current_player, board, bag)

    def end_game(self):
        """Forces the game to end when the bag runs out of tiles or too many skipped turns."""
        print("\nGame Over!")
        
        # Deduct points for remaining tiles in rack
        for player in self.players:
            deduction = 0
            for tile in player.rack.rack:
                if tile in LETTER_VALUES:
                    deduction += LETTER_VALUES[tile]
            player.increase_score(-deduction)
            if deduction > 0:
                print(f"{player.get_name()} loses {deduction} points for unplayed tiles.")

        # Find winner
        highest_score = -1
        winning_player = None
        all_player_scores = {}
        
        for player in self.players:
            all_player_scores[player.get_name()] = player.get_score()
            if player.get_score() > highest_score:
                highest_score = player.get_score()
                winning_player = player.get_name()
        
        # Display results
        print(f"\nThe game is over! {winning_player} has won with {highest_score} points!")
        for player_name, score in all_player_scores.items():
            if player_name != winning_player:
                print(f"{player_name} ended the game with {score} points!")

def main():
    dict_file_path = "build/scrabbledict.txt"
    global dictionary
    dictionary = load_dictionary_from_file(dict_file_path)
    scrabble = Game(dictionary)
    ai_game = input("Would you like to play against an AI? (y/n): ").strip().lower() == 'y'
    if ai_game:
        dict_file_path = "scrabbledict.txt"
        ai_num = input("Do you want AI vs AI? (y/n): ").strip().lower() == 'y'
        if ai_num:
            scrabble.start_ai_game()
        else:
            scrabble.start_game_vs_ai()
    else: 
        scrabble.start_game()


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     from word_generator import load_dictionary, get_possible_words

#     # Example placeholder objects for standalone testing
#     class FakeBoard:
#         def __init__(self):
#             self.grid = [[" " for _ in range(15)] for _ in range(15)]

#     class FakePlayer:
#         def __init__(self, rack):
#             self.rack = rack

#     board = FakeBoard()
#     player = FakePlayer(['S', 'T', 'A', 'R', 'E', 'L', 'D'])

#     dicttrie = load_dictionary_from_file("scrabbledict.txt")

#     dictionary = load_dictionary("scrabbledict.txt")
#     legal_moves = get_possible_words(board, player.rack, dictionary, player, Word)

#     initial_state = {
#         'board': board,
#         'rack': player.rack,
#         'legal_moves': legal_moves,
#         'player': player  # ← Add this
#     }

#     best_move = monte_carlo_tree_search(initial_state, iterations=500)
#     print("Best Move Found:", best_move)
