from dictionarytrie import DictionaryTrie
from player import Player
from word import Word
import math
import random
import copy

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

class ScrabbleAI(Player):
    def __init__(self, dictionary, board, bag, strategy):
        super().__init__(bag)  # Initialize the Player attributes
        self.dict = dictionary
        self.board = board

    def set_strat(self, strat_name):
        if strat_name in ["BEAM", "ASTAR", "GBFS", "BFS"]:
            self.name = f"AI_{strat_name}"
        else:
            self.name = "AI_MCTS"
        
        self.name = f"AI_{strat_name}"

    def get_name(self):
        #Gets the AIplayer's name.
        return self.name

    def make_move(self):
        """Make the best move according to the current strategy"""
        # Find all legal moves
        legal_moves = find_all_moves(self.board, self.rack)
        
        if not legal_moves:
            return None, "skip"
        
        # Use the strategy to find the best move
        # USE a "find_best_move" function WITHIN EACH STRAT
        best_move = self.get_best_move(self.board, self.rack, legal_moves)
        
        if not best_move:
            return None, "skip"
        
        return best_move, "play"
    
    def get_best_move(self):
        """
        Get the best move based on the selected strategy.
        
        Returns:
            The best move according to the selected strategy
        """
        if self.name == "AI_GBFS":
            # Greedy best first search strategy
            return self.get_gbfs_move()
        elif self.name == "AI_ASTAR":
            # A* strategy
            return self.get_astar_move()
        elif self.name == "AI_BEAM":
            # Beam strategy
            return self.get_beam_move()
        elif self.name == "AI_BFS":
            # BFS strategy
            return self.get_bfs_move()
        else:
            # Default to Monte Carlo Tree Search strategy
            return self.get_mcts_move()
    
    def get_gbfs_move(self):
        """TO BE IMPLEMENTED"""
        return
    
    def get_astar_move(self):
        """TO BE IMPLEMENTED"""
        return
    
    def get_beam_move(self):
        """TO BE IMPLEMENTED"""
        return
    
    def get_bfs_move(self):
        """TO BE IMPLEMENTED"""
        return
    
    def get_mcts_move(self):
        """TO BE IMPLEMENTED"""
        return
    
    ######### SCRABBLE AI ^^^ | vvv Global funcs below ####################################
    
def get_beam_move(dictionary, board, rack, legal_moves):

    if not dictionary or not rack or not legal_moves or not board:
        return None, ""
    valid_moves = find_all_moves(board, rack)
    move_tree = create_word_tree(valid_moves, rack)
    best_move = beam_search(move_tree, 10)

    
def create_word_tree(moves, rack):
    move_tree = DictionaryTrie()
    tile_score = 0
    for word, pos, dir, placed, score in moves:
        check = move_tree.get_node(word)
        new_score = score - rack_score(placed, rack)
        if check and check.score < new_score:
            check.set_attr(word, pos, dir, new_score)
        else:
            curr = move_tree.add_word(word)
            curr.set_attr(word, pos, dir, new_score)
    return move_tree


def rack_score(placed, rack):
    score = 0
    con = 0
    vow = 0
    rack_arr = []
    for tile in rack:
        rack_arr.append(tile.get_char())
        
    for location, letter in placed:
        if letter in rack_arr:
            rack_arr.remove(letter)
            if letter == "#":
                score += 1
            elif letter in ['A', 'E', 'I', 'O', 'U']:
                vow += 1
            else:
                con += 1
    
    diff = con - vow
    if abs(diff) <= 1:
        score += 2
    elif abs(diff) >= 5:
        score -= 3
    elif abs(diff) >= 4:
        score -= 2
    elif abs(diff) >= 3:
        score -= 1
    
    return score

def beam_search(move_tree, beam_width=10, beam_depth):
    """
    Perform beam search on the word tree to find the best move.
    
    The beam search algorithm works by:
    1. Starting at the root of the tree
    2. Evaluating all child nodes
    3. Keeping only the top N (beam_width) candidates
    4. Continuing this process until leaf nodes are reached
    5. Returning the highest scoring move found
    
    Args:
        move_tree: DictionaryTrie containing moves
        beam_width: Number of candidates to keep at each level
        
    Returns:
        dict: Best move information or None if no moves found
    """
    # Start at the root node
    root = move_tree.root
    
    # Initialize the beam with the root node
    beam = [root]
    
    # Best move found so far
    best_move = None
    best_score = 0
    
    # Continue until the beam is empty
    while beam:
        # Collect all children of nodes in the current beam
        candidates = []
        
        for node in beam:
            # If this node represents a complete word with attributes
            if node.word and node.score:
                # Check if this is the best move found so far
                if node.score > best_score:
                    best_score = node.score
                    best_move = {
                        'word': node.word,
                        'position': node.position,
                        'direction': node.direction,
                        'score': node.score
                    }
            
            # Add all children to candidates
            for letter, child_node in node.children.items():
                candidates.append(child_node)
        
        # If no candidates, we've reached the end of the tree
        if not candidates:
            break
        
        # Sort candidates by score (if available) or depth in the tree
        candidates.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        
        # Keep only the top beam_width candidates
        beam = candidates[:beam_width]
    
    return best_move
    
def find_anchor_points(board):

    """Find all empty cells adjacent to placed tiles."""
    """Returns a set of anchor locations (row, col) on board"""
    anchor_points = set()
    current_node = board.start_node
    
    # Traverse the board to find anchor points
    for i in range(board.size):
        row_node = current_node
        for j in range(board.size):
            if not row_node.tile:
                # Check if any adjacent node is occupied
                if ((row_node.right and row_node.right.tile) or
                    (row_node.left and row_node.left.tile) or
                    (row_node.up and row_node.up.tile) or
                    (row_node.down and row_node.down.tile)):
                    anchor_points.add(row_node.position)
            row_node = row_node.right
        current_node = current_node.down
    return list(anchor_points)
    
def get_cross_checks(board, dict, row, col, direction):
    """
    Calculate which letters can be legally placed at a position based on
    cross-checks (words formed in the perpendicular direction).
    
    Args:
        row, col: Position to check
        direction: 'across' or 'down'
    
    Returns:
        Set of valid letters that can be placed at this position
    """
    valid_letters = set(LETTER_VALUES)  # Start with all letters
    
    node = board.get_node(row, col)
    if not node:
        return set()  # Out of bounds
        
    if node.tile:
        # If the square is already occupied, only the existing letter is valid
        return {node.tile}
    
    # If this is an empty square, check what cross-words would be formed
    if direction == "right":
        # Check for vertical constraints (words formed top-to-bottom)
        # If there's no tiles above or below, all letters are valid
        node_up = board.get_node(row-1, col)
        node_down = board.get_node(row+1, col)
        
        if not (node_up and node_up.tile) and not (node_down and node_down.tile):
            return valid_letters
        
        # There's at least one adjacent tile vertically, so we need to check what vertical
        # words would be formed
        
        # Find the start of the potential vertical word
        start_row = row
        while True:
            prev_row = start_row - 1
            if prev_row >= 0:
                prev_node = board.get_node(prev_row, col)
                if prev_node and prev_node.tile:
                    start_row = prev_row
                else:
                    break
            else:
                break
        
        # For each possible letter, check if it forms a valid vertical word
        valid_cross_letters = set()
        
        for letter in valid_letters:
            # Build the vertical word with this letter
            word = ""
            curr_row = start_row
            
            while curr_row < board.size:
                curr_node = board.get_node(curr_row, col)
                if not curr_node:
                    break
                
                if curr_row == row:
                    # This is our test position
                    word += letter
                elif curr_node.tile:
                    word += curr_node.tile.letter
                else:
                    break
                
                curr_row += 1
            
            # If the word is just this letter (no constraints), all letters are valid
            if word == letter:
                valid_cross_letters.add(letter)
                continue
            
            # Check if this is a valid word in our dictionary
            if dict.is_word(word):
                valid_cross_letters.add(letter)
        
        return valid_cross_letters
    
    else:  # direction == 'down'
        # Similar logic for horizontal constraints when placing vertically
        node_left = board.get_node(row, col-1)
        node_right = board.get_node(row, col+1)
        
        if not (node_left and node_left.tile) and not (node_right and node_right.tile):
            return valid_letters
        
        # Find the start of the potential horizontal word
        start_col = col
        while True:
            prev_col = start_col - 1
            if prev_col >= 0:
                prev_node = board.get_node(row, prev_col)
                if prev_node and prev_node.tile:
                    start_col = prev_col
                else:
                    break
            else:
                break
        
        # For each possible letter, check if it forms a valid horizontal word
        valid_cross_letters = set()
        
        for letter in valid_letters:
            # Build the horizontal word with this letter
            word = ""
            curr_col = start_col
            
            while curr_col < board.size:
                curr_node = board.get_node(row, curr_col)
                if not curr_node:
                    break
                
                if curr_col == col:
                    # This is our test position
                    word += letter
                elif curr_node.tile:
                    word += curr_node.tile.letter
                else:
                    break
                
                curr_col += 1
            
            # If the word is just this letter (no constraints), all letters are valid
            if word == letter:
                valid_cross_letters.add(letter)
                continue
            
            # Check if this is a valid word in our dictionary
            if dict.is_word(word):
                valid_cross_letters.add(letter)
        
        return valid_cross_letters

def find_all_moves(board, rack):
    """
    Find all valid moves for the current board and rack.
    
    Returns:
        List of valid moves with scores
    """
    valid_moves = []
    anchor_points = board.find_anchor_points()
    
    # Find moves for each anchor point in both directions
    for row, col in anchor_points:
        # Try horizontal placement
        right_moves = find_moves_at_anchor(row, col, rack, "right")
        
        # Try vertical placement
        down_moves = find_moves_at_anchor(row, col, rack, "down")
    
    valid_moves = right_moves + down_moves
    
    # Sort moves by score (highest first)
    return valid_moves

def find_moves_at_anchor(board, anchor_row, anchor_col, rack, direction):
    """
    Find all valid moves that go through a specific anchor point in a given direction.
    
    Args:
        anchor_row, anchor_col: The anchor position
        rack: Available letters
        direction: 'right' or 'down'
    """
    valid_moves = []
    # Find the maximum prefix length (how far we can go before the anchor)
    prefix_limit = calculate_prefix_limit(board, anchor_row, anchor_col, direction)
    
    # Try each possible prefix length
    for prefix_length in range(prefix_limit + 1):
        # Calculate the starting position for this prefix length
        if direction == "right":
            start_row, start_col = anchor_row, anchor_col - prefix_length
        else:  # direction == 'down'
            start_row, start_col = anchor_row - prefix_length, anchor_col
        
        # Find all words that can be placed starting at this position
        generate_moves(start_row, start_col, rack, direction, prefix_length, board, valid_moves) ############### what to set this to?
    
    return valid_moves

def calculate_prefix_limit(board, row, col, direction):
    """
    Calculate the maximum number of tiles that can be placed before an anchor point.
    
    Args:
        row, col: The anchor position
        direction: 'right' or 'down'
    
    Returns:
        Maximum number of tiles that can be placed before the anchor
    """
    limit = 0
    
    if direction == "right":
        # Count empty squares to the left
        curr_col = col - 1
        while curr_col >= 0:
            node = board.get_node(row, curr_col)
            if node and not node.tile:
                limit += 1
                curr_col -= 1
            else:
                break
    else:  # direction == 'down'
        # Count empty squares above
        curr_row = row - 1
        while curr_row >= 0:
            node = board.get_node(curr_row, col)
            if node and not node.tile:
                limit += 1
                curr_row -= 1
            else:
                break
    
    return limit

def generate_moves(dict, start_row, start_col, rack, direction, prefix_length, board, valid_moves):
    """
    Generate all possible moves starting from a specific position.
    
    Args:
        start_row, start_col: Starting position
        rack: Available letters
        direction: 'right' or 'down'
        prefix_length: How far into the word is the anchor point
    """
    # Start with the root of the dictionary trie
    generate_moves_recursive("", dict.root, start_row, start_col, board, 
                                    rack, direction, prefix_length, [], False, valid_moves)

def generate_moves_recursive(partial_word, dict_node, row, col, available_rack, board, 
                            direction, remaining_prefix, placed_tiles, word_has_anchor, valid_moves):
    """
    Recursively generate all valid moves starting from a position.
    
    Args:
        partial_word: Word built so far
        dict_node: Current node in the dictionary trie
        row, col: Current position on the board
        available_rack: Letters still available in the rack
        direction: 'right' or 'down'
        remaining_prefix: How many more letters before reaching the anchor
        placed_tiles: List of tiles placed so far [(position, letter)]
        word_has_anchor: Whether the word uses an existing anchor point
    """
    # Check if we're still on the board
    node = board.get_node(row, col)
    if not node:
        # We've gone off the board, so check if we have a valid word
        if dict_node.is_terminal and word_has_anchor and partial_word:
            # We have a complete word that uses an anchor
            record_move(partial_word, placed_tiles, direction, valid_moves)
        return
    
    # If this square is already occupied, we must use that letter
    if node.tile:
        # Get the letter on this square
        letter = node.tile.char
        
        # Check if this letter continues a valid path in our dictionary
        next_node = dict_node.get_child(letter)
        if next_node:
            # This letter is valid, so continue building the word
            
            # Calculate next position
            next_row, next_col = get_next_position(row, col, direction)
            
            # Continue recursively
            generate_moves_recursive(
                partial_word + letter,
                next_node,
                next_row,
                next_col,
                available_rack,
                board,
                direction,
                0,  # No more prefix needed since we've already placed at least one tile
                placed_tiles + [((row, col), None)],  # Mark that we used an existing tile
                True,  # We've now used at least one anchor
                valid_moves
            )
    else:
        # The square is empty, we can place any letter from our rack
        
        # If we need to place a prefix tile, we don't check cross-constraints yet
        if remaining_prefix > 0:
            valid_letters = set(available_rack)
        else:
            # Get the set of valid letters based on cross-checks
            valid_letters = get_cross_checks(board, row, col, direction).intersection(available_rack)
        
        # Try each valid letter
        for tile in valid_letters:
            letter = tile.get_letter()
            # Check if this letter continues a valid path in our dictionary
            next_node = dict_node.get_child(letter)
            if next_node:
                # This letter is valid, so continue building the word
                
                # Remove the letter from the rack
                rack_copy = available_rack.copy()
                rack_copy.remove(letter)
                
                # Calculate next position
                next_row, next_col = get_next_position(row, col, direction)
                
                # Continue recursively
                generate_moves_recursive(
                    partial_word + letter,
                    next_node,
                    next_row,
                    next_col,
                    rack_copy,
                    board,
                    direction,
                    max(0, remaining_prefix - 1),
                    placed_tiles + [((row, col), letter)],
                    word_has_anchor or remaining_prefix == 0, # A tile at the anchor counts
                    valid_moves
                )
            elif letter == "#":
                rack_copy = available_rack.copy()
                rack_copy.remove(letter)
                
                # Calculate next position
                next_row, next_col = get_next_position(row, col, direction)
                
                for char in dict_node.children:
                    # Continue recursively
                    generate_moves_recursive(
                        partial_word + char,
                        next_node,
                        next_row,
                        next_col,
                        rack_copy,
                        board,
                        direction,
                        max(0, remaining_prefix - 1),
                        placed_tiles + [((row, col), char)],
                        word_has_anchor or remaining_prefix == 0,  # A tile at the anchor counts
                        valid_moves
                )
        
        # If we have a valid word so far and we've used an anchor, record it
        if dict_node.is_terminal and word_has_anchor and partial_word:
            ### ADD TO List of valid moves
            record_move(partial_word, placed_tiles, direction, valid_moves, board)

def get_next_position(row, col, direction):
    """Get the next position based on the current direction."""
    if direction == "right":
        return row, col + 1
    else:  # direction == 'down'
        return row + 1, col

def record_move(word, placed_tiles, direction, valid_moves, board):
    """
    Record a valid move in the list of valid moves.
    
    Args:
        word: The word formed
        placed_tiles: List of tiles placed [(position, letter)]
        direction: 'right' or 'down'
    """
    # Calculate score for this placement
    score = calculate_placement_score(placed_tiles, board, direction)
    
    # Get the starting position of the word
    if placed_tiles:
        start_pos = min(placed_tiles, key=lambda x: x[0] if direction == 'down' else x[0][1])[0]
    else:
        return  # No tiles placed
    
    # Add the move to our list of valid moves
    move = {
        'word': word,
        'start_position': start_pos,
        'direction': direction,
        'placed_tiles': placed_tiles,
        'score': score
    }
    
    valid_moves.append(move)

def calculate_placement_score(placed_tiles, board, direction):
    """
    Calculate the score for a set of placed tiles.
    
    Args:
        placed_tiles: List of tiles placed [(position, letter)]
        
    Returns:
        Total score for this placement
    """
    total_score = 0
    main_word_score = 0
    main_word_multiplier = 1
    tiles_used = 0
    
    # Track the main word being formed (in the placement direction)
    main_word_tiles = []
    
    # First pass: calculate the main word score and any cross-word scores
    for position, letter in placed_tiles:
        row, col = position
        node = board.get_node(row, col)
        
        if not node:
            continue
            
        # If this is a tile we're placing (not an existing one)
        if letter is not None:
            letter_score = LETTER_VALUES[letter]
            tiles_used += 1
            
            # Apply letter multipliers
            if node.score_multiplier == "TLS":
                letter_score *= 3
            elif node.score_multiplier == "DLS":
                letter_score *= 2
            
            # Track word multipliers for the main word
            if node.score_multiplier == "TWS":
                main_word_multiplier *= 3
            elif node.score_multiplier == "DWS":
                main_word_multiplier *= 2
            
            main_word_score += letter_score
            main_word_tiles.append((position, letter))
            
            # Check for cross-words formed
            cross_word_score = calculate_cross_word_score(row, col, letter, direction)
            total_score += cross_word_score
        else:
            # This is an existing tile we're using
            letter = node.tile
            letter_score = LETTER_VALUES[letter]
            main_word_score += letter_score
            main_word_tiles.append((position, letter))
    
    # Apply the word multiplier to the main word
    total_score += (main_word_score * main_word_multiplier)
    
    # Bonus for using all 7 tiles
    if tiles_used == 7:
        total_score += 50
        
    return total_score
    
def calculate_cross_word_score(board, row, col, letter, direction):
    """
    Calculate the score for any cross-word formed by placing a letter.
    
    Args:
        row, col: Position of the placed letter
        letter: The letter being placed
        direction: 'right' or 'down' - the main direction of word placement
        
    Returns:
        Score for any cross-word formed, or 0 if none
    """
    # If we're placing horizontally, check for vertical cross-words
    # If we're placing vertically, check for horizontal cross-words
    cross_direction = 'down' if direction == "right" else 'right'
    
    # Check if this letter forms a cross-word
    # A cross-word is formed if there are adjacent tiles in the perpendicular direction
    if cross_direction == 'down':
        # Check for tiles above or below
        if not (board.get_node(row-1, col) and board.get_node(row-1, col).tile) and \
            not (board.get_node(row+1, col) and board.get_node(row+1, col).tile):
            return 0  # No cross-word formed
    else:  # cross_direction == 'right'
        # Check for tiles to the left or right
        if not (board.get_node(row, col-1) and board.get_node(row, col-1).tile) and \
            not (board.get_node(row, col+1) and board.get_node(row, col+1).tile):
            return 0  # No cross-word formed
    
    # A cross-word is formed, calculate its score
    word = ""
    word_multiplier = 1
    word_score = 0
    
    # Find the start of the cross-word
    if cross_direction == 'down':
        # Find the topmost tile of the vertical word
        curr_row = row
        while curr_row > 0:
            upper_node = board.get_node(curr_row-1, col)
            if upper_node and upper_node.tile:
                curr_row -= 1
            else:
                break
        
        # Build the word from top to bottom
        start_row = curr_row
        while True:
            curr_node = board.get_node(curr_row, col)
            if not curr_node:
                break
            
            if curr_row == row:
                # This is where we're placing our new letter
                curr_letter = letter
                letter_score = LETTER_VALUES[letter]
                
                # Apply letter multipliers
                if curr_node.score_multiplier == "TLS":
                    letter_score *= 3
                elif curr_node.score_multiplier == "DLS":
                    letter_score *= 2
                
                # Track word multipliers
                if curr_node.score_multiplier == "TWS":
                    word_multiplier *= 3
                elif curr_node.score_multiplier == "DWS":
                    word_multiplier *= 2
            elif curr_node.tile:
                curr_letter = curr_node.tile
                letter_score = LETTER_VALUES[curr_letter]
            else:
                break
            
            word += curr_letter
            word_score += letter_score
            curr_row += 1
    
    else:  # cross_direction == 'right'
        # Find the leftmost tile of the horizontal word
        curr_col = col
        while curr_col > 0:
            left_node = board.get_node(row, curr_col-1)
            if left_node and left_node.tile:
                curr_col -= 1
            else:
                break
        
        # Build the word from left to right
        start_col = curr_col
        while True:
            curr_node = board.get_node(row, curr_col)
            if not curr_node:
                break
            
            if curr_col == col:
                # This is where we're placing our new letter
                curr_letter = letter
                letter_score = LETTER_VALUES[letter]
                
                # Apply letter multipliers
                if curr_node.score_multiplier == "TLS":
                    letter_score *= 3
                elif curr_node.score_multiplier == "DLS":
                    letter_score *= 2
                
                # Track word multipliers
                if curr_node.score_multiplier == "TWS":
                    word_multiplier *= 3
                elif curr_node.score_multiplier == "DWS":
                    word_multiplier *= 2
            elif curr_node.tile:
                curr_letter = curr_node.tile
                letter_score = LETTER_VALUES[curr_letter]
            else:
                break
            
            word += curr_letter
            word_score += letter_score
            curr_col += 1
    
    # Only count as a cross-word if it's at least 2 letters long
    if len(word) >= 2:
        return word_score * word_multiplier
    return 0



class MCTS:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move  # e.g., {'word': ..., 'position': ..., 'direction': ..., 'score': ...}
        self.children = []
        self.visits = 0
        self.total_score = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state['legal_moves'])

    def best_child(self, c_param=1.4):
        return max(
            self.children,
            key=lambda c: (c.total_score / (c.visits + 1e-4)) + c_param * ((2 * math.log(self.visits + 1)) / (c.visits + 1e-4))**0.5
        )

def rollout(state):
    """Simulate placing a random legal move using game logic, and return the resulting score."""
    legal_moves = state['legal_moves']
    if not legal_moves:
        return 0

    move = random.choice(legal_moves)
    temp_board = state['board']
    temp_player = state['player']

    # Create the Word object and validate the move
    
    word_obj = Word(move['word'], list(move['position']), temp_player, move['direction'], temp_board)
    if word_obj.check_word() is True:
        # Simulate scoring
        temp_player.score = 0
        word_obj.calculate_word_score()
        score = temp_player.get_score()
        temp_player.score = 0  # Reset score after simulation
        return score

    return 0  # Invalid move

def expand(node):
    """Add a new child node from the list of untried moves."""
    tried_moves = [child.move for child in node.children]
    untried_moves = [m for m in node.state['legal_moves'] if m not in tried_moves]
    if not untried_moves:
        return node
    move = random.choice(untried_moves)
    new_state = copy.deepcopy(node.state)
    new_state['legal_moves'].remove(move)
    child = MCTS(new_state, parent=node, move=move)
    node.children.append(child)
    return child

def backpropagate(node, result):
    while node:
        node.visits += 1
        node.total_score += result
        node = node.parent

def monte_carlo_tree_search(initial_state, iterations=1000):
    root = MCTS(initial_state)

    for _ in range(iterations):
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        leaf = expand(node)
        result = rollout(leaf.state)
        backpropagate(leaf, result)

    best = max(root.children, key=lambda c: c.visits)
    return best.move


