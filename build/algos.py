from dictionarytrie import DictionaryTrie
from player import Player, Tile
import random
from collections import deque
import heapq
import math
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
    def __init__(self, bag, dictionary, board, strategy):
        super().__init__(bag, dictionary, board)  # Initialize the Player attributes
        self.set_strat(strategy)

    def set_strat(self, strat_name):
        if strat_name in ["BEAM", "ASTAR", "GBFS", "BFS", "DFS", "UCS"]:
            self.name = f"AI_{strat_name}"
        else:
            self.name = "AI_MCTS"

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

    def deep_copy_player(self, player, new_board, new_bag):
        """
        Create a deep copy of a Player object, connecting it to the new board and bag.
        """
        if isinstance(player, ScrabbleAI):
            # Create a new AI player
            new_player = ScrabbleAI(
                player.dict,  # Dictionary can be shared as it doesn't change
                new_board,
                new_bag,
                player.name.split('_')[1] if '_' in player.name else "MCTS"
            )
        else:
            new_player = Player(
                player.dict,  # Dictionary can be shared as it doesn't change
                new_board,
                new_bag,
                player.name.split('_')[1] if '_' in player.name else "MCTS"
            )
       
        # Copy the rack
        new_player.rack = copy.deepcopy(player.rack)
       
        # Copy score
        if hasattr(player, 'score'):
            new_player.score = player.score
           
        return new_player
    
    def get_best_move(self):
        """
        Get the best move based on the selected strategy.
        
        Returns:
            The best move according t
            the selected strategy
        """
        if self.name == "AI_ASTAR":
            # A* strategy
            return get_astar_move(self.board, self.rack, self.dict)
        elif self.name == "AI_BFS":
            # BFS strategy
            return get_bfs_move(self.board, self.rack, self.dict)
        elif self.name == "AI_DFS":
            # dFS strategy
            return get_dfs_move(self.board, self.rack, self.dict)
        elif self.name == "AI_UCS":
            # dFS strategy
            return get_dijkstra_move(self.board, self.rack, self.dict)
        elif self.name == "AI_GBFS":
            #GBFS strategy
            return get_greedy_move(self.board, self.rack, self.dict)
        else: # MCTS strategy
            # Default to Monte Carlo Tree Search strategy
            return get_mcts_move(self.board, self.rack, self.dict)
    
    ######### SCRABBLE AI ^^^ | vvv Global funcs below ####################################

def get_mcts_move(board, rack, dict):
    valid_moves = find_all_moves(board, rack, dict)
    if not valid_moves:
        return None

    # Set up root node
    root_state = {
        'board': board,
        'rack': rack,
        'legal_moves': valid_moves
    }
    root = Node(root_state)
    iterations = 100  # or 500, or whatever number you want


    for _ in range(iterations):
        node = root

        # Selection phase
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # Expansion phase
        leaf = expand(node)

        # Simulation phase
        result = rollout(leaf.state)

        # Backpropagation phase
        backpropagate(leaf, result)

    # Choose best move from root's children
    best_child = max(root.children, key=lambda c: c.visits)
    move = best_child.move
    return [move[0], move[1], move[2]]  # word, pos, direction

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move  # (word, pos, direction, placed, score)
        self.children = []
        self.visits = 0
        self.total_score = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state['legal_moves'])

    def best_child(self, c_param=1.4):
        return max(
            self.children,
            key=lambda c: (c.total_score / (c.visits + 1e-4)) + c_param * math.sqrt(math.log(self.visits + 1) / (c.visits + 1e-4))
        )

def expand(node):
    tried_moves = [child.move for child in node.children]
    untried_moves = [m for m in node.state['legal_moves'] if m not in tried_moves]
    if not untried_moves:
        return node

    move = random.choice(untried_moves)
    new_state = copy.deepcopy(node.state)
    new_state['legal_moves'].remove(move)

    child = Node(new_state, parent=node, move=move)
    node.children.append(child)
    return child

def rollout(state):
    legal_moves = state['legal_moves']
    if not legal_moves:
        return 0

    move = random.choice(legal_moves)
    _, _, _, _, score = move  # unpack score directly from the move tuple
    return score

def backpropagate(node, result):
    while node:
        node.visits += 1
        node.total_score += result
        node = node.parent
        
### MCTS ^^^
    
def get_bfs_move(board, rack, dict):
    valid_moves = find_all_moves(board, rack, dict)
    move_tree = create_word_tree(valid_moves, rack, False)

    return bfs_search(move_tree.root)

def bfs_search(node):
    best_move = None
    best_score = -1
    # Recursively search through all children
    queue = deque([node])

    while queue:
        node = queue.popleft()

        if hasattr(node, 'is_terminal') and node.is_terminal:
            if hasattr(node, 'score') and node.score > best_score:
                best_score = node.score
                best_move = [node.word, node.position, node.direction]

        for child_node in node.children.values():
            queue.append(child_node)

    return best_move

###

def get_dfs_move(board, rack, dict):
    valid_moves = find_all_moves(board, rack, dict)
    move_tree = create_word_tree(valid_moves, rack, False)

    return dfs_search(move_tree.root)

def dfs_search(node):
    best_move = None
    best_score = -1
    
    stack = [node]
    
    while stack:
        node = stack.pop()
    
        if hasattr(node, 'is_terminal') and node.is_terminal:
            if hasattr(node, 'score') and node.score > best_score:
                best_score = node.score
                best_move = [node.word, node.position, node.direction]

        # Add children to the stack
        for child_node in node.children.values():
            stack.append(child_node)
        
    return best_move

###

def get_astar_move(board, rack, dict):
    valid_moves = find_all_moves(board, rack, dict)
    move_tree = create_word_tree(valid_moves, rack, True)
    
    return astar_search(move_tree.root)

def astar_search(node):
    best_move = None
    best_score = -1
    
    # Use a counter to break ties in the priority queue
    counter = 0
    
    # Priority queue stores tuples of (-score, counter, node)
    # Using negative score because heapq is a min-heap but we want max score
    pq = [(-node.score if hasattr(node, 'score') else 0, counter, node)]
    counter += 1
    
    while pq:
        # Get node with highest priority (highest score)
        _, _, current_node = heapq.heappop(pq)
        
        # Check if this is a terminal node with a valid move
        if hasattr(current_node, 'is_terminal') and current_node.is_terminal:
            if hasattr(current_node, 'score') and current_node.score > best_score:
                best_score = current_node.score
                best_move = [current_node.word, current_node.position, current_node.direction]
        
        # Add all children to the priority queue
        for child_node in current_node.children.values():
            # Only consider nodes that have a score attribute
            if hasattr(child_node, 'score'):
                heapq.heappush(pq, (-child_node.score, counter, child_node))
                counter += 1
    
    return best_move

###
    
def get_dijkstra_move(board, rack, dict): ## same thing as Uniform Cost Search (UCS)
    valid_moves = find_all_moves(board, rack, dict)
    move_tree = create_word_tree(valid_moves, rack, True)  # Include scores in the tree

    return dijkstra_search(move_tree.root)

def dijkstra_search(node):
    """
    Implementation of Dijkstra's algorithm for finding the optimal move in Scrabble.
    
    In this context, Dijkstra is used to find the move with the highest score,
    treating lower scores as "longer distances".
    
    Args:
        node: The root node of the move tree
        
    Returns:
        The best move [word, position, direction]
    """
    best_move = None
    best_score = -1
    
    # Dictionary to keep track of the highest score found for each node
    scores = {node: 0}
    
    # Use a counter to break ties and avoid comparing Node objects directly
    counter = 0
    
    # Priority queue stores tuples of (-score, counter, node)
    # Using negative score because heapq is a min-heap but we want max score
    pq = [(0, counter, node)]
    counter += 1
    
    # Set to keep track of processed nodes
    processed = set()
    
    while pq:
        # Get node with highest priority (highest score)
        current_neg_score, _, current_node = heapq.heappop(pq)
        current_score = -current_neg_score  # Convert back to positive score
        
        # Skip if we've already processed this node
        if current_node in processed:
            continue
            
        processed.add(current_node)
        
        # Check if this is a terminal node with a valid move
        if hasattr(current_node, 'is_terminal') and current_node.is_terminal:
            if hasattr(current_node, 'score') and current_node.score > best_score:
                best_score = current_node.score
                best_move = [current_node.word, current_node.position, current_node.direction]
        
        # Process all neighbors (children)
        for child_node in current_node.children.values():
            if child_node not in processed:
                # Calculate the score for this path
                # In Dijkstra, we're looking for highest score, not lowest distance
                child_score = current_score
                if hasattr(child_node, 'score'):
                    child_score += child_node.score
                
                # If we found a better score to reach this node
                if child_node not in scores or child_score > scores[child_node]:
                    scores[child_node] = child_score
                    heapq.heappush(pq, (-child_score, counter, child_node))
                    counter += 1
    
    return best_move

###

def get_greedy_move(board, rack, dict):
    valid_moves = find_all_moves(board, rack, dict)
    if not valid_moves:
        return None
    
    move_tree = create_word_tree(valid_moves, rack, False)
    return gbfs(move_tree.root)
    
def gbfs(node):
    """
    Implementation of Greedy Best First Search for finding a move in Scrabble.
    
    Unlike Dijkstra's algorithm which considers the cumulative score along a path,
    GBFS only considers the heuristic value (the score of each individual move).
    
    Args:
        node: The root node of the move tree
    
    Returns:
        The best move [word, position, direction] according to GBFS
    """
    best_move = None
    best_score = -1
    
    # Use a counter to break ties and ensure consistent behavior
    counter = 0
    
    # Priority queue stores tuples of (-heuristic_value, counter, node)
    # Using negative heuristic because heapq is a min-heap but we want max score
    pq = [(0, counter, node)]
    counter += 1
    
    # Set to keep track of visited nodes
    visited = set()
    
    while pq:
        # Get node with highest priority (highest heuristic value)
        current_neg_score, _, current_node = heapq.heappop(pq)
        
        # Skip if we've already visited this node
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # Check if this is a terminal node with a valid move
        if hasattr(current_node, 'is_terminal') and current_node.is_terminal:
            # In GBFS, we only consider the heuristic value (score) of the move itself
            if hasattr(current_node, 'score') and current_node.score > best_score:
                best_score = current_node.score
                best_move = [current_node.word, current_node.position, current_node.direction]
        
        # Process all neighbors (children)
        for child_node in current_node.children.values():
            if child_node not in visited:
                # For GBFS, we only consider the heuristic value (score) of the node itself
                # We don't accumulate scores from the parent
                child_score = 0
                if hasattr(child_node, 'score'):
                    child_score = child_node.score
                
                heapq.heappush(pq, (-child_score, counter, child_node))
                counter += 1
    
    return best_move

###

def get_beam_move(board, rack, player, bag, dict, beam_width=3, search_depth=2):
    """
    Find the best move using beam search.
    
    Args:
        board: Current game board
        rack: Current player's letter rack
        dict: Dictionary of valid words
        beam_width: Number of top states to keep at each level
        search_depth: How many moves to look ahead
        
    Returns:
        Best move as [word, position, direction]
    """
    # Get all valid moves from the current state
    valid_moves = find_all_moves(board, rack, dict)
    if not valid_moves:
        return None
       
    # Initial beam contains just the starting point
    # Instead of storing full copies, store the move sequence and score
    initial_state = {
        'moves_made': [],
        'cumulative_score': 0
    }
   
    # Use a list of states as our beam
    current_beam = [initial_state]
   
    # Cache all found moves to avoid recalculation
    # Format: (board_hash, rack_hash) -> list of valid moves
    move_cache = {}
    
    # For each depth level
    for depth in range(search_depth):
        next_beam = []
        print(f"Searching depth {depth+1}/{search_depth}, beam size: {len(current_beam)}")
       
        # Expand each state in the current beam
        for state_idx, state in enumerate(current_beam):
            # We need to reconstruct the current state by replaying moves
            current_board = board.copy()  # Shallow copy
            current_player = player.copy()  # Shallow copy
            current_bag = bag.copy()  # Shallow copy
            
            # Replay all moves to get to this state
            for move in state['moves_made']:
                word, position, direction, placed_tiles, move_score = move
                
                # Apply move to current_board
                apply_move_to_board_fast(current_board, word, position, direction, placed_tiles)
                
                # Update player's rack and score (without deep copies)
                update_player_after_move_fast(current_player, current_bag, placed_tiles, move_score)
            
            # Generate a hash key for the current board and rack state
            board_hash = hash_board(current_board)
            rack_hash = hash_rack(current_player.rack)
            cache_key = (board_hash, rack_hash)
            
            # Check if we've already calculated moves for this state
            if cache_key in move_cache:
                moves = move_cache[cache_key]
            else:
                # Get valid moves from this state
                moves = find_all_moves(current_board, current_player.rack, dict)
                move_cache[cache_key] = moves
                
            if not moves:
                # If no moves possible, keep this state in the beam
                next_beam.append(state)
                continue
               
            # For each move, create a new state (without deep copies)
            for move in moves:
                word, position, direction, placed_tiles, move_score = move
                
                # Create a new state that references the previous move sequence
                new_state = {
                    'moves_made': state['moves_made'] + [move],
                    'cumulative_score': state['cumulative_score'] + move_score
                }
                
                # Add to candidates for next beam
                next_beam.append(new_state)
       
        # Keep only the top beam_width states based on cumulative score
        current_beam = heapq.nlargest(beam_width, next_beam,
                                      key=lambda s: s['cumulative_score'])
       
    if not current_beam:
        return None
       
    best_state = max(current_beam, key=lambda s: s['cumulative_score'])
   
    # Return the first move from the best path
    if best_state['moves_made']:
        first_move = best_state['moves_made'][0]
        return [first_move[0], first_move[1], first_move[2]]  # word, pos, direction
   
    return None

    
def create_word_tree(moves, rack, h):
    move_tree = DictionaryTrie()
    first_word = ""
    for word, pos, dir, placed, score in moves:
        if h:
            score = score + rack_score(placed, rack)
        check = move_tree.get_node(word)
        if check and check.is_terminal and check.score < score:
            check.set_attr(word, pos, dir, score)
        elif not check:
            curr = move_tree.add_word(word)
            curr.set_attr(word, pos, dir, score)    
    return move_tree

def rack_score(placed, rack): # used in astar search
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
                score += 3
            elif letter in ['A', 'E', 'I', 'O', 'U']:
                vow += 1
                total_rack_score -= 1
            else:
                con += 1
                total_rack_score -= LETTER_VALUES[letter]
        i = i - 1
    
    diff = con - vow
    if abs(diff) <= 1:
        score += 4
    elif abs(diff) >= 5:
        score -= 6
    elif abs(diff) >= 4:
        score -= 4
    elif abs(diff) >= 3:
        score -= 2
        
    for tile in new_rack:
        if (diff >= 3) and (tile in ['Z', 'Q']):
            score -= 5

    return score


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

def simulateGame():
    return None
    
def find_anchor_points(board):
    """
    Find all empty cells adjacent to placed tiles.
    Returns a set of anchor locations (row, col) on board
    """
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
    """
    Get the next position based on the current direction.
    """
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

def deep_copy_bag(bag):
    """
    Create a deep copy of the Bag object.
    """
    return copy.deepcopy(bag)

def deep_copy_player(player, dict, new_board, new_bag):
    """
    Create a deep copy of a Player object, connecting it to the new board and bag.
    """
    # tiles_available = []
    # tiles = player.get_rack_arr()
    # for tile in tiles:
    #     tiles_available = tiles_available + tile.get_letter()
    
    # for tile in new_bag.bag:
    #     tiles_available = tiles_available + tile.get_letter()
        
    # print(tiles_available)
    
    if isinstance(player, ScrabbleAI):
        # Create a new AI player
        new_player = ScrabbleAI(
            new_bag,  # Dictionary can be shared as it doesn't change
            dict,
            new_board,
            player.name.split('_')[1] if '_' in player.name else "MCTS"
        )
    # else:
    #     # For regular players
    #     new_player = Player(new_bag)
    #     new_player.name = player.name
    
    # Copy the rack
    new_player.rack = copy.deepcopy(player.rack)
    
    # Copy score
    if hasattr(player, 'score'):
        new_player.score = player.score
        
    return new_player

def apply_move_to_board(board, word, position, direction, placed_tiles):
    """
    Apply a move to the board.
    
    Args:
        board: The ScrabbleBoard object
        word: The word to place
        position: (row, col) tuple for word start
        direction: 'across' or 'down'
        placed_tiles: List of tiles being placed
    """
    row, col = position
    
    # Place each letter on the board
    for i, letter in enumerate(word):
        if direction == 'across':
            current_pos = (row, col + i)
        else:  # direction == 'down'
            current_pos = (row + i, col)
            
        # Get the node at this position
        node = board._get_node_at_position(board.start_node, current_pos)
        
        # If this node doesn't already have a tile (it's one we're placing)
        if node and not node.tile:
            # Find the correct tile from placed_tiles
            for _, tile in placed_tiles:
                row, col = current_pos
                curr_node = board.get_node(row, col)
                if tile == letter:
                    used_tile = Tile(tile, LETTER_VALUES)
                    curr_node.place_tile(used_tile)
                elif tile == "#":
                    used_tile = Tile("#", LETTER_VALUES)
                    curr_node.place_blank(used_tile, letter)
                
def update_player_after_move(player, bag, placed_tiles, move_score):
    """
    Update the player's rack and score after making a move.
    
    Args:
        player: Player object to update
        bag: Bag object to draw from
        placed_tiles: List of tiles placed on the board
        move_score: Score from the move
    """
    # Remove placed tiles from rack
    for tile in placed_tiles:
        if tile in player.rack.rack:
            player.rack.remove(tile)
    
    # Draw new tiles if available
    while len(player.rack.rack) < 7 and len(bag.tiles) > 0:
        player.rack.replenish_rack()
    
    # Update score
    player.score += move_score
    
def hash_board(board):
    """
    Create a hash representation of the board for caching.
    Only considers tile placements, not the full board object.
    
    Args:
        board: The ScrabbleBoard object
    
    Returns:
        A hashable representation of the board state
    """
    # Create a tuple of tuples with occupied positions and their values
    occupied_positions = []
    
    # Iterate through the board's nodes
    current_node = board.start_node
    while current_node:
        row_node = current_node
        while row_node:
            if row_node.tile:
                # Store position and letter
                pos = (row_node.row, row_node.col)
                letter = row_node.get_display_letter()
                occupied_positions.append((pos, letter))
            row_node = row_node.right
        current_node = current_node.down
    
    # Convert to a tuple for hashability
    return tuple(sorted(occupied_positions))

def hash_rack(rack):
    """Create a hashable representation of a rack"""
    # Sort the tiles for consistent hashing
    return tuple(sorted(str(tile) for tile in rack.rack))

def apply_move_to_board_fast(board, word, position, direction, placed_tiles):
    """
    Apply a move to the board without deep copying.
    Optimized version that modifies the board in-place.
    
    Args:
        board: The ScrabbleBoard object
        word: The word to place
        position: (row, col) tuple for word start
        direction: 'across' or 'down'
        placed_tiles: List of tiles being placed
    """
    row, col = position
    
    # Create a map of positions to placed tiles for quick lookup
    placed_positions = {}
    for i, letter in enumerate(word):
        if direction == 'across':
            current_pos = (row, col + i)
        else:  # direction == 'down'
            current_pos = (row + i, col)
        
        # Get the node at this position
        node = board._get_node_at_position(board.start_node, current_pos)
        
        # If this node doesn't already have a tile (it's one we're placing)
        if node and not node.tile:
            placed_positions[current_pos] = letter
    
    # Now place each tile
    for tile_info in placed_tiles:
        tile, pos = tile_info  # Assuming placed_tiles contains (tile, position) pairs
        if pos in placed_positions:
            curr_node = board.get_node(pos[0], pos[1])
            
            if tile == "#":  # Blank tile
                used_tile = Tile("#", LETTER_VALUES)
                curr_node.place_blank(used_tile, placed_positions[pos])
            else:
                used_tile = Tile(tile, LETTER_VALUES)
                curr_node.place_tile(used_tile)
                
def update_player_after_move_fast(player, bag, placed_tiles, move_score):
    """
    Update the player's rack and score after making a move.
    Optimized version that modifies the player and bag in-place.
    
    Args:
        player: Player object to update
        bag: Bag object to draw from
        placed_tiles: List of tiles placed on the board
        move_score: Score from the move
    """
    # Remove placed tiles from rack
    for tile_info in placed_tiles:
        tile = tile_info[0]  # Assuming placed_tiles contains (tile, position) pairs
        if tile in player.rack.rack:
            player.rack.remove(tile)
    
    # Draw new tiles if available
    while len(player.rack.rack) < 7 and len(bag.tiles) > 0:
        drawn_tile = bag.draw_tile()
        if drawn_tile:
            player.rack.add(drawn_tile)
    
    # Update score
    player.score += move_score
