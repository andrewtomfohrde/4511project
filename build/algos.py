from dictionarytrie import DictionaryTrie
from player import Player
from word import Word
import math
import random
import copy
from collections import deque

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
        self.set_strat(strategy)

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
        # Use the strategy to find the best move
        # USE a "find_best_move" function WITHIN EACH STRAT
        best_move = self.get_best_move()
        
        if not best_move:
            return None, "sk1p"
        
        return best_move, "play"
    
    def get_best_move(self):
        """
        Get the best move based on the selected strategy.
        
        Returns:
            The best move according to the selected strategy
        """
        if self.name == "AI_GBFS":
            # Greedy best first search strategy
            return get_gbfs_move()
        elif self.name == "AI_ASTAR":
            # A* strategy
            return get_astar_move()
        elif self.name == "AI_BEAM":
            # Beam strategy
            return get_beam_move(self.board, self.rack)
        elif self.name == "AI_BFS":
            # BFS strategy
            return get_bfs_move(self.board, self.rack, self.dict)
        elif self.name == "AI_DFS":
            # dFS strategy
            return get_dfs_move(self.board, self.rack, self.dict)
        else:
            # Default to Monte Carlo Tree Search strategy
            return get_mcts_move(self.board, self.rack, self.dict)
    
    def get_gbfs_move(self):
        """TO BE IMPLEMENTED"""
        return
    
    def get_astar_move(self):
        """TO BE IMPLEMENTED"""
        return
    
    ######### SCRABBLE AI ^^^ | vvv Global funcs below ####################################

def get_mcts_move(board, rack, dict):
    from copy import deepcopy
    import random
    import math

    def best_child(node_children, parent_visits, c_param=1.4):
        def uct(child):
            exploitation = child['total_score'] / (child['visits'] + 1e-4)
            exploration = c_param * math.sqrt(math.log(parent_visits + 1) / (child['visits'] + 1e-4))
            return exploitation + exploration
        return max(node_children, key=uct)

    def rollout(state):
        legal_moves = state['legal_moves']
        if not legal_moves:
            return 0
        move = random.choice(legal_moves)
        word_obj = Word(move[0], list(move[1]), state['player'], move[2], state['board'])
        if word_obj.check_word():
            state['player'].score = 0
            word_obj.calculate_word_score(move[3])
            score = state['player'].get_score()
            state['player'].score = 0
            return score
        return 0

    def expand(node):
        tried = [child['move'] for child in node['children']]
        untried = [m for m in node['state']['legal_moves'] if m not in tried]
        if not untried:
            return node
        move = random.choice(untried)
        new_state = deepcopy(node['state'])
        new_state['legal_moves'].remove(move)
        child_node = {
            'state': new_state,
            'move': move,
            'visits': 0,
            'total_score': 0,
            'children': [],
            'parent': node
        }
        node['children'].append(child_node)
        return child_node

    def backpropagate(node, result):
        while node:
            node['visits'] += 1
            node['total_score'] += result
            node = node['parent']

    valid_moves = find_all_moves(board, rack, dict)
    if not valid_moves:
        return None

    # Create a dummy player with valid bag (copied from the rack)
    dummy_player = Player(rack.bag)  # Use same bag
    dummy_player.rack = rack.copy()  # Copy the rack

    root_state = {
        'board': board,
        'player': dummy_player,
        'legal_moves': valid_moves.copy()
    }
    root = {
        'state': root_state,
        'move': None,
        'visits': 0,
        'total_score': 0,
        'children': [],
        'parent': None
    }

    for _ in range(250):  # Adjustable simulation count
        node = root
        while node['children'] and len(node['children']) == len(node['state']['legal_moves']):
            node = best_child(node['children'], node['visits'])

        leaf = expand(node)
        result = rollout(leaf['state'])
        backpropagate(leaf, result)

    if not root['children']:
        return None

    best = max(root['children'], key=lambda c: c['visits'])
    word, pos, direction = best['move'][0], best['move'][1], best['move'][2]
    return [word, pos, direction]



    
def get_bfs_move(board, rack, dict):
    valid_moves = find_all_moves(board, rack, dict)
    best_move = None

    if valid_moves:
        word, pos, dir, placed, score = valid_moves[0]
        best_move = [word, pos, dir]

    return best_move

def get_dfs_move(board, rack, dict):
    valid_moves = find_all_moves(board, rack, dict)
    move_tree = create_word_tree(valid_moves, rack)

    return dfs_search(move_tree.root)

def dfs_search(node):
    best_move = None
    best_score = -1
    
    if hasattr(node, 'is_terminal') and node.is_terminal:
        best_move = [node.word, node.position, node.direction]
        best_score = node.score
    
    # Recursively search through all children
    for letter, child_node in node.children.items():
        child_result = dfs_search(child_node)
        
        # If child_result is not None, it's a list [word, position, direction]
        # The score is stored in the node
        if child_result and hasattr(child_node, 'score'):
            child_score = child_node.score
            if child_score > best_score:
                best_score = child_score
                best_move = child_result
    
    return best_move


def get_astar_move(board, rack, dict):
    valid_moves = find_all_moves(board, rack, dict)
    move_tree = create_word_tree(valid_moves, rack)



def get_beam_move(board, rack):

    if not rack or not board:
        return None, ""
    
    valid_moves = find_all_moves(board, rack)
    if not valid_moves:
        return None, ""
    move_tree = create_word_tree(valid_moves, rack)

    best_move = beam_search(move_tree, 10)
    return best_move

    
def create_word_tree(moves, rack):
    move_tree = DictionaryTrie()
    first_word = ""
    for word, pos, dir, placed, score in moves:
        check = move_tree.get_node(word)
        new_score = score + rack_score(placed, rack)
        if check and check.is_terminal and check.score < new_score:
            check.set_attr(word, pos, dir, new_score)
        elif not check:
            curr = move_tree.add_word(word)
            curr.set_attr(word, pos, dir, new_score)
    
    return move_tree


def rack_score(placed, rack):
    score = 0
    con = 0
    vow = 0
    new_rack = []
    total_rack_score = 0
    rack_arr = rack.get_rack_arr()
    i = 0
    for tile in rack_arr:
        el = tile.get_letter()
        new_rack.append(el)
        total_rack_score += LETTER_VALUES[el]
        i = i + 1

    for location, letter in placed:
        if letter in new_rack:
            new_rack.remove(letter)
            if letter == "#":
                score += 1
            elif letter in ['A', 'E', 'I', 'O', 'U']:
                vow += 1
                total_rack_score -= 1
            else:
                con += 1
                total_rack_score -= LETTER_VALUES[letter]
    
    diff = con - vow
    if abs(diff) <= 1:
        score += 2
    elif abs(diff) >= 5:
        score -= 3
    elif abs(diff) >= 4:
        score -= 2
    elif abs(diff) >= 3:
        score -= 1
    
    score += total_rack_score
    
    return score

def beam_search(move_tree, beam_width=10):
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
    
    depth = 0
    
    # Continue until the beam is empty
    while beam:
        depth+=1
        # Collect all children of nodes in the current beam
        candidates = []
        
        for node in beam:
            # If this node represents a complete word with attributes
            if node.word and node.score:
                # Check if this is the best move found so far
                if node.score > best_score:
                    best_move = [
                        node.word,
                        node.position,
                        node.direction,
                    ]
                    best_score = node.score
            
            # Add all children to candidates
            for letter, child_node in node.children.items():
                candidates.append(child_node)
        
        # If no candidates, we've reached the end of the tree
        if not candidates:
            break
        
        # Sort candidates by our multi-criteria scoring function
        candidates.sort(key=candidate_score, reverse=True)
        
        # Keep only the top beam_width candidates
        beam = candidates[:beam_width]
    
    return best_move
    
def candidate_score(node):
    # Primary criterion: actual score (if available)
    score = getattr(node, 'score', 0) or 0
    
    # Secondary criteria for nodes without complete scores
    secondary_score = 0
    
    # Consider terminal nodes (complete words) more valuable
    if getattr(node, 'is_terminal', False):
        secondary_score += 10000  # High value to prioritize complete words
    
    # Consider nodes with more children (more potential) as more promising
    child_count = len(getattr(node, 'children', {}))
    secondary_score += child_count * 100
    
    # Consider depth in the search to prefer longer words
    if getattr(node, 'word', ''):
        secondary_score += len(node.word) * 10
    
    return (score, secondary_score)
    
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
        
    if not anchor_points:
        center = board.get_node(7,7)
        anchor_points.add(center.position)
    
    return list(anchor_points)
    

    
def get_cross_checks(board, dict, row, col, direction):
    """
    Calculate which letters can be legally placed at a position based on
    cross-checks (words formed in the perpendicular direction).
    
    Args:
        board: The game board
        dict: Dictionary of valid words
        row, col: Position to check
        direction: 'right' for horizontal words, 'down' for vertical words
    
    Returns:
        Set of valid letters that can be placed at this position
    """
    # Start with all possible letters (assuming LETTER_VALUES is defined elsewhere)
    # Replace this with the appropriate set of letters for your game
    valid_letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  
    
    node = board.get_node(row, col)
    if not node:
        return set()  # Out of bounds
        
    if node.tile:
        # If the square is already occupied, only the existing letter is valid
        return {node.tile.get_letter()}
    
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
                    word += curr_node.tile.get_letter()
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
                    word += curr_node.tile.get_letter()
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

def find_all_moves(board, rack, dict):
    """
    Find all valid moves for the current board and rack.
    
    Returns:
        List of valid moves with scores
    """
    valid_moves = []
    anchor_points = find_anchor_points(board) 
    # Find moves for each anchor point in both directions
    for row, col in anchor_points:
        # Try horizontal placement
        right_moves = find_moves_at_anchor(dict, board, row, col, rack, "right")
        # print(f"RIGHT: {right_moves}")
        if right_moves:
            valid_moves = valid_moves + right_moves
         
        # Try vertical placement
        down_moves = find_moves_at_anchor(dict, board, row, col, rack, "down")
        # print(f"DOWN: {down_moves}")
        if down_moves:
            valid_moves = valid_moves + down_moves
    
    # print(f"VALID: {valid_moves}")
    # Sort moves by score (highest first)
    return valid_moves

def find_moves_at_anchor(dict, board, anchor_row, anchor_col, rack, direction):
    """
    Find all valid moves that go through a specific anchor point in a given direction.
   
    Args:
        dict: The dictionary containing valid words
        board: The current game board
        anchor_row, anchor_col: The anchor position
        rack: Rack object containing available letters
        direction: 'right' or 'down'
        
    Returns:
        List of valid moves that can be placed through the anchor point
    """
    valid_moves = []
    
    placed = []
    
    # Find the maximum prefix length (how far we can go before the anchor)
    prefix_limit = calculate_prefix_limit(board, anchor_row, anchor_col, direction)
   
    # Try each possible prefix length
    for prefix_length in range(prefix_limit + 1):
        # Calculate the starting position for this prefix length
        if direction == "right":
            start_row, start_col = anchor_row, anchor_col - prefix_length
            
            # Check if there are existing tiles before our starting position
            # and adjust start position if necessary
            prefix_letters = ""
            i = 1
            while True:
                check = board.get_node(start_row, start_col - i)
                if check and check.tile:
                    prefix_letters = check.tile.get_letter() + prefix_letters
                    placed = [((start_row, start_col - i), None)] + placed
                    i += 1
                else:
                    break
                    
        else:  # direction == 'down'
            start_row, start_col = anchor_row - prefix_length, anchor_col
            
            # Check if there are existing tiles before our starting position
            # and adjust start position if necessary
            prefix_letters = ""
            i = 1
            while True:
                check = board.get_node(start_row - i, start_col)
                if check and check.tile:
                    prefix_letters = check.tile.get_letter() + prefix_letters
                    placed = [((start_row - i, start_col), None)] + placed
                    i += 1
                else:
                    break
       
        # Find all words that can be placed starting at this position
        dict_node = dict.root
        # Process any existing prefix letters to get the right dictionary node
        for letter in prefix_letters:
            dict_node = dict_node.get_child(letter)
            if not dict_node:
                break
            
        rack_copy = []
        nurack = rack.get_rack_arr()
        for tile in nurack:
            rack_copy.append(tile.get_letter())
            
        if dict_node:  # Only proceed if the prefix is valid
            # Call the recursive function to generate moves
            generate_moves_recursive(
                prefix_letters,
                dict,
                dict_node,
                start_row,
                start_col,
                board,
                rack_copy,  # Pass a copy of the rack to avoid modifying the original
                direction,
                prefix_length,
                placed,
                False,  # Will become true when we place/use a tile at the anchor
                valid_moves
            )
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
        while curr_row >= 0 and not board.get_node(curr_row, col).tile:
            node = board.get_node(curr_row, col)
            if node and not node.tile:
                limit += 1
                curr_row -= 1
            else:
                break
    
    return min(limit, 6)   
    
def generate_moves_recursive(partial_word, dictionary, dict_node, row, col, board, rack,
                          direction, remaining_prefix, placed_tiles, word_has_anchor, valid_moves):
    """
    Recursively generate all valid moves starting from a position.
    
    Args: 
        partial_word: Word built so far
        dictionary: The dictionary trie
        dict_node: Current node in the dictionary trie
        row, col: Current position on the board
        board: Game board
        rack: Rack object containing available letters
        direction: 'right' or 'down'
        remaining_prefix: How many more letters before reaching the anchor
        placed_tiles: List of tiles placed so far [(position, letter)]
        word_has_anchor: Whether the word uses an existing anchor point
        valid_moves: List to store valid moves
    """
    # Check if we're still on the board
    node = board.get_node(row, col)
    next_row, next_col = get_next_position(row, col, direction)
    if not node:
        # We've gone off the board, so check if we have a valid word
        if dict_node.is_terminal and word_has_anchor and partial_word and placed_tiles:
            # We have a complete word that uses an anchor
            next_node = board.get_node(next_row, next_col)
            if not next_node or not next_node.tile:
                record_move(partial_word, placed_tiles, direction, valid_moves, board)    
        return
    
    # Calculate next position for future recursion
    
    
    # If this square is already occupied, we must use that letter
    if node.tile:
        # Get the letter on this square
        letter = node.tile.get_letter()
        # Check if this letter continues a valid path in our dictionary
        next_node = dict_node.get_child(letter)
        if next_node:
            # This letter is valid, so continue building the word
            new_placed_tiles = placed_tiles + [((row, col), None)]  # None indicates existing tile
            
            # Continue recursively
            generate_moves_recursive(
                partial_word + letter,
                dictionary,
                next_node,
                next_row,
                next_col,
                board,
                rack.copy(),  # Pass a copy of the rack to prevent side effects
                direction,
                max(0, remaining_prefix - 1),
                new_placed_tiles,
                True,  # We've now used at least one anchor
                valid_moves
            )
    else:
        # The square is empty, we can place any letter from our rack
        
        # If we need to place a prefix tile, we can use any tile from rack
        valid_letters = get_cross_checks(board, dictionary, row, col, direction)
        
        # For each tile in the rack
        rack_copy = rack.copy()  # Work with a copy to avoid modifying the original
        for i, letter in enumerate(rack_copy):
            
            # Skip if this letter isn't valid for cross-checks
            if letter != "#" and letter not in valid_letters:
                continue
                
            # Make a new rack without this tile
            new_rack = rack_copy.copy()
            new_rack.pop(i)
            
            if letter == "#":  # Blank tile - can represent any letter
                # Try each possible letter from valid cross-checks
                for char in valid_letters:
                    next_node = dict_node.get_child(char)
                    if next_node:
                        new_placed_tiles = placed_tiles + [((row, col), letter)]
                        
                        # Continue recursively with this letter
                        generate_moves_recursive(
                            partial_word + char,
                            dictionary,
                            next_node,
                            next_row,
                            next_col,
                            board,
                            new_rack,
                            direction,
                            max(0, remaining_prefix - 1),
                            new_placed_tiles,
                            word_has_anchor or remaining_prefix==0,
                            valid_moves
                        )
            else:  # Regular tile
                next_node = dict_node.get_child(letter)
                if next_node:
                    new_placed_tiles = placed_tiles + [((row, col), letter)]
                    
                    # Continue recursively
                    generate_moves_recursive(
                        partial_word + letter,
                        dictionary,
                        next_node,
                        next_row,
                        next_col,
                        board,
                        new_rack,
                        direction,
                        max(0, remaining_prefix - 1),
                        new_placed_tiles,
                        word_has_anchor or remaining_prefix==0,
                        valid_moves
                    )
    
    # Check if current position is the end of a valid word
    # We need this check here to catch words that end at this position
    if dict_node.is_terminal and word_has_anchor and partial_word:
        # Check if we can end the word here (no continuing tiles in direction)
        next_node = board.get_node(next_row, next_col)
        if not next_node or not next_node.tile:
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
        valid_moves: List to store valid moves
        board: Game board for scoring
    """
    if valid_moves is None:
        valid_moves = []
    
    # Filter out None tiles (existing tiles on the board)
    start_pos = placed_tiles[0][0]
    
    # Calculate score
    score = calculate_placement_score(placed_tiles, board, direction)
    
    # Create move record
    move = [
        word,
        start_pos,
        direction,
        placed_tiles,
        score
    ]
    
    # Add to list of valid moves
    valid_moves.append(move)
    return valid_moves

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
    
    # First pass: calculate the main word score and any cross-word scores
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
    for tile in placed_tiles:
        sec = False
        sec_score = 0
        xls = 1
        sxws = 1
        location, letter = tile
        if location:
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
            if direction == "right":
                while curr_node.up and curr_node.up.tile:
                    curr_node = curr_node.up
                while curr_node.tile:
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
                while curr_node.tile:
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

# class MCTS:
#     def __init__(self, state, parent=None, move=None):
#         self.state = state
#         self.parent = parent
#         self.move = move  # e.g., {'word': ..., 'position': ..., 'direction': ..., 'score': ...}
#         self.children = []
#         self.visits = 0
#         self.total_score = 0

#     def is_fully_expanded(self):
#         return len(self.children) == len(self.state['legal_moves'])

#     def best_child(self, c_param=1.4):
#         return max(
#             self.children,
#             key=lambda c: (c.total_score / (c.visits + 1e-4)) + c_param * ((2 * math.log(self.visits + 1)) / (c.visits + 1e-4))**0.5
#         )

# def rollout(state):
#     """Simulate placing a random legal move using game logic, and return the resulting score."""
#     legal_moves = state['legal_moves']
#     if not legal_moves:
#         return 0

#     move = random.choice(legal_moves)
#     temp_board = state['board']
#     temp_player = state['player']

#     # Create the Word object and validate the move
    
#     word_obj = Word(move['word'], list(move['position']), temp_player, move['direction'], temp_board)
#     if word_obj.check_word() is True:
#         # Simulate scoring
#         temp_player.score = 0
#         word_obj.calculate_word_score()
#         score = temp_player.get_score()
#         temp_player.score = 0  # Reset score after simulation
#         return score

#     return 0  # Invalid move

# def expand(node):
#     """Add a new child node from the list of untried moves."""
#     tried_moves = [child.move for child in node.children]
#     untried_moves = [m for m in node.state['legal_moves'] if m not in tried_moves]
#     if not untried_moves:
#         return node
#     move = random.choice(untried_moves)
#     new_state = copy.deepcopy(node.state)
#     new_state['legal_moves'].remove(move)
#     child = MCTS(new_state, parent=node, move=move)
#     node.children.append(child)
#     return child

# def backpropagate(node, result):
#     while node:
#         node.visits += 1
#         node.total_score += result
#         node = node.parent

# def monte_carlo_tree_search(initial_state, iterations=1000):
#     root = MCTS(initial_state)

#     for _ in range(iterations):
#         node = root
#         while node.is_fully_expanded() and node.children:
#             node = node.best_child()
#         leaf = expand(node)
#         result = rollout(leaf.state)
#         backpropagate(leaf, result)

#     best = max(root.children, key=lambda c: c.visits)
#     return best.move


