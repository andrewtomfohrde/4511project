from random import shuffle
import re
from dictionarytrie import DictionaryTrie
from word import Word, load_dictionary_from_file
from algos import ScrabbleAI
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
            self.position = (0, 0) # (row, column)
            self.score_multiplier = ""
            # TRIPLE_WORD_SCORE = ((0,0), (7, 0), (14,0), (0, 7), (14, 7), (0, 14), (7, 14), (14,14))
            # DOUBLE_WORD_SCORE = ((1,1), (2,2), (3,3), (4,4), (1, 13), (2, 12), (3, 11), (4, 10), (7, 7), (13, 1), (12, 2), (11, 3), (10, 4), (13,13), (12, 12), (11,11), (10,10))
            # TRIPLE_LETTER_SCORE = ((1,5), (1, 9), (5,1), (5,5), (5,9), (5,13), (9,1), (9,5), (9,9), (9,13), (13, 5), (13,9))
            # DOUBLE_LETTER_SCORE = ((0, 3), (0,11), (2,6), (2,8), (3,0), (3,7), (3,14), (6,2), (6,6), (6,8), (6,12), (7,3), (7,11), (8,2), (8,6), (8,8), (8, 12), (11,0), (11,7), (11,14), (12,6), (12,8), (14, 3), (14, 11))
            
        
        def place_tile(self, tile):
            self.tile = tile
            
        def place_blank(self, tile, char):
            self.tile = tile
            self.tile.char = char
            
        def get_display_str(self):
            """Return a string representation of this cell."""
            if self.tile:
                return f"{self.tile.char}/{self.tile.letter}"
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
                for tile in player.rack.get_rack_arr():
                    # For blank tiles, look for '#'
                    if is_blank and tile.get_letter() == '#':
                        used_tile = tile
                        node.place_blank(used_tile, placed_tiles[i][1])
                        print(f"Placing tile # as {placed_tiles[i][1]}")
                        break
                    # For regular tiles, look for matching letter
                    elif not is_blank and tile.get_letter() == letter:
                        used_tile = tile
                        node.place_tile(used_tile)
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


##############################################################################3###


class Game:
    def __init__(self):
        self.round_number = 1
        self.players = []
        self.skipped_turns = 0
        self.board = ScrabbleBoard()
        self.bag = Bag()
        self.init_dict()

    def add_player(self, player):
        """Add a player to the game"""
        self.players.append(player)

    def init_dict(self):
        dict_file_path = "build/scrabbledict.txt"
        dict = load_dictionary_from_file(dict_file_path)
        self.dictionary = dict

    def turn(self, player):
        """
        Handle a player's turn
        """
        if (self.skipped_turns < 6) or (player.rack.get_rack_length() == 0 and self.bag.get_remaining_tiles() == 0):
            print(f"\nRound {self.round_number}: {player.get_name()}'s turn \n")
            print(self.board.get_board())
            print(f"\n{player.get_name()}'s Letter Rack: {player.get_rack_str()}")

            # Check if the player is an AI
            if isinstance(player, ScrabbleAI):
                print(f"[{player.get_name()} is thinking...]")

                best_move, action = player.make_move()

                
                # Get AI's move
                if player.get_name() in ["AI_MCTS", "AI_BEAM"]:
                    player.make_move()
                else:
                    best_move, action = player.make_move()
                
                if action == "sk1p" and not best_move:
                    print(f"{player.get_name()} has no valid moves. Skipping turn.")
                    self.skipped_turns += 1
                else:
                    word_to_play = best_move[0]
                    location = best_move[1]
                    direction = best_move[2]
                    
                    print(f"{player.get_name()} plays: {word_to_play} at {location} going {direction}")
                    
                    # Create and validate the word
                    word = Word(word_to_play, location, player, direction, self.board)
                    valid, placed = word.check_word()
                    
                    if valid:
                        self.board.place_word(word_to_play, location, direction, player, placed)
                        word_score = word.calculate_word_score(placed)
                        print(f"Word '{word.word}' placed for {word_score} points!")
                        player.increase_score(word_score)
                        self.skipped_turns = 0
                    else:
                        print(f"{player.get_name()} attempted invalid word. Skipping.")
                        self.skipped_turns += 1

                input("Continue? :")
            
            # Human player's turn
            else:
                placed = []
                checked = False
                while not checked:
                    word_to_play = input("Word to play: ")
                    
                    # Check if player wants to skip turn
                    if word_to_play.lower() == "":
                        print(f"{player.get_name()} skips their turn.")
                        self.skipped_turns += 1
                        checked = True
                        break
                        
                    location = []
                    col = input("Column number: ")
                    row = input("Row number: ")
                    if (col == "" or row == "") or (col not in [str(x) for x in range(15)] or row not in [str(x) for x in range(15)]):
                        location = [-1, -1]
                    else:
                        location = [int(row), int(col)]
                    direction = input("Direction of word (right or down): ")

                    word = Word(word_to_play, location, player, direction, self.board)

                    # Handle blank tiles
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

                    checked, placed = word.chdicteck_word()
                
                if checked and not word_to_play.lower() in "":
                    success = self.board.place_word(word_to_play, location, direction, player, placed)
                    word_score = word.calculate_word_score(placed)
                    if success:
                        print(f"Word '{word.word}' placed for {word_score} points!")
                        player.increase_score(word_score)
                        self.skipped_turns = 0
                    else:
                        print("Failed to place word. Skipping turn.")
                        self.skipped_turns += 1

            print(f"\n{player.get_name()}'s score is: {player.get_score()}")

            # Determine next player
            if self.players.index(player) != (len(self.players) - 1):
                next_player = self.players[self.players.index(player) + 1]
            else:
                next_player = self.players[0]
                self.round_number += 1
            player.set_curr(False)
            next_player.set_curr(True)

            # Refill player's rack
            player.rack.replenish_rack()
            
            # Recursive call for next player's turn
            self.turn(next_player)
        else:
            self.end_game()

    def start_game(self):
        """Start a game with human players only"""
        # Create players
        num_players = int(input("Enter number of players (2-4): "))
        for i in range(min(num_players, 4)):
            player = Player(self.bag)
            player.set_name(input(f"Enter name for player {i+1}: "))
            self.add_player(player)
        
        # Start the game with the first player
        self.turn(self.players[0])

    def start_game_vs_ai(self):
        """Start a game with one human player and one AI player"""
        # Create human player
        human = Player(self.bag)
        human.set_name(input("Enter your name: "))
        self.add_player(human)
        
        # Create AI player
        ai_strategy = input("Select AI strategy (MCTS/BEAM/ASTAR/GBFS/BFS/DFS/UCS): ").upper()
        ai = ScrabbleAI(self.dictionary, self.board, self.bag, ai_strategy)
        self.add_player(ai)
        
        # Start the game with the human player
        self.turn(self.players[0])

    def start_ai_game(self):
        """Start a game with AI players only"""
        # Create AI players
        num_ais = int(input("Enter number of AI players (2-4): "))
        for i in range(min(num_ais, 4)):
            ai_strategy = input(f"Select strategy for AI {i+1} (MCTS/Beam/ASTAR/GBFS/BFS/DFS/UCS): ").upper()
            ai = ScrabbleAI(self.dictionary, self.board, self.bag, ai_strategy)
            self.add_player(ai)
        
        # Start the game with the first AI
        self.turn(self.players[0])

    def end_game(self):
        """Forces the game to end when the bag runs out of tiles or too many skipped turns."""
        print("\nGame Over!")
        
        # Deduct points for remaining tiles in rack
        for player in self.players:
            deduction = 0
            for tile in player.rack.get_rack_arr():
                if tile in LETTER_VALUES:
                    deduction += LETTER_VALUES[tile.get_letter()]
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
    game = Game()
    
    print("Welcome to Scrabble!")
    game_type = input("Select game type:\n1. Human vs Human\n2. Human vs AI\n3. AI vs AI\nChoice: ")
    
    if game_type == "1":
        game.start_game()
    elif game_type == "2":
        game.start_game_vs_ai()
    elif game_type == "3":
        game.start_ai_game()
    else:
        print("Invalid choice. Starting Human vs Human game by default.")
        game.start_game()


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
