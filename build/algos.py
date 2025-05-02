from dictionarytrie import DictionaryTrie
from player import Player, Rack

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
    def __init__(self, name, dictionary, board, bag, strategy):
        super().__init__(bag)  # Initialize the Player attributes
        self.dictionary = dictionary
        self.board = board
        self.strategy = strategy
        self.valid_moves = []
        self.direction = None
    
    def get_node_at(self, row, col):
        """
        Get the board node at the specified position.
        
        Args:
            row, col: Position coordinates
            
        Returns:
            BoardNode at the position or None if out of bounds
        """
        try:
            return self.board.get_node(row, col)
        except (IndexError, AttributeError):
            return None
    
    def get_cross_checks(self, row, col, direction):
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
        
        node = self.get_node_at(row, col)
        if not node:
            return set()  # Out of bounds
            
        if node.occupied:
            # If the square is already occupied, only the existing letter is valid
            return {node.tile}
        
        # If this is an empty square, check what cross-words would be formed
        if direction == 'across':
            # Check for vertical constraints (words formed top-to-bottom)
            # If there's no tiles above or below, all letters are valid
            node_up = self.get_node_at(row-1, col)
            node_down = self.get_node_at(row+1, col)
            
            if not (node_up and node_up.occupied) and not (node_down and node_down.occupied):
                return valid_letters
            
            # There's at least one adjacent tile vertically, so we need to check what vertical
            # words would be formed
            
            # Find the start of the potential vertical word
            start_row = row
            while True:
                prev_row = start_row - 1
                if prev_row >= 0:
                    prev_node = self.get_node_at(prev_row, col)
                    if prev_node and prev_node.occupied:
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
                
                while curr_row < self.board.size:
                    curr_node = self.get_node_at(curr_row, col)
                    if not curr_node:
                        break
                    
                    if curr_row == row:
                        # This is our test position
                        word += letter
                    elif curr_node.occupied:
                        word += curr_node.tile
                    else:
                        break
                    
                    curr_row += 1
                
                # If the word is just this letter (no constraints), all letters are valid
                if word == letter:
                    valid_cross_letters.add(letter)
                    continue
                
                # Check if this is a valid word in our dictionary
                if self.dictionary.is_word(word):
                    valid_cross_letters.add(letter)
            
            return valid_cross_letters
        
        else:  # direction == 'down'
            # Similar logic for horizontal constraints when placing vertically
            node_left = self.get_node_at(row, col-1)
            node_right = self.get_node_at(row, col+1)
            
            if not (node_left and node_left.occupied) and not (node_right and node_right.occupied):
                return valid_letters
            
            # Find the start of the potential horizontal word
            start_col = col
            while True:
                prev_col = start_col - 1
                if prev_col >= 0:
                    prev_node = self.get_node_at(row, prev_col)
                    if prev_node and prev_node.occupied:
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
                
                while curr_col < self.board.size:
                    curr_node = self.get_node_at(row, curr_col)
                    if not curr_node:
                        break
                    
                    if curr_col == col:
                        # This is our test position
                        word += letter
                    elif curr_node.occupied:
                        word += curr_node.tile
                    else:
                        break
                    
                    curr_col += 1
                
                # If the word is just this letter (no constraints), all letters are valid
                if word == letter:
                    valid_cross_letters.add(letter)
                    continue
                
                # Check if this is a valid word in our dictionary
                if self.dictionary.is_word(word):
                    valid_cross_letters.add(letter)
            
            return valid_cross_letters
    
    def find_all_moves(self):
        """
        Find all valid moves for the current board and rack.
        
        Returns:
            List of valid moves with scores
        """
        self.valid_moves = []
        anchor_points = self.board.find_anchor_points()
        
        # Find moves for each anchor point in both directions
        for row, col in anchor_points:
            # Try horizontal placement
            self.direction = "right"
            self.find_moves_at_anchor(row, col, self.rack, 'right')
            
            # Try vertical placement
            self.direction = "down"
            self.find_moves_at_anchor(row, col, self.rack, 'down')
        
        # Sort moves by score (highest first)
        self.valid_moves.sort(key=lambda move: move['score'], reverse=True)
        return self.valid_moves
    
    def find_moves_at_anchor(self, anchor_row, anchor_col, rack, direction):
        """
        Find all valid moves that go through a specific anchor point in a given direction.
        
        Args:
            anchor_row, anchor_col: The anchor position
            rack: Available letters
            direction: 'right' or 'down'
        """
        # Find the maximum prefix length (how far we can go before the anchor)
        prefix_limit = self.calculate_prefix_limit(anchor_row, anchor_col, direction)
        
        # Try each possible prefix length
        for prefix_length in range(prefix_limit + 1):
            # Calculate the starting position for this prefix length
            if direction == 'right':
                start_row, start_col = anchor_row, anchor_col - prefix_length
            else:  # direction == 'down'
                start_row, start_col = anchor_row - prefix_length, anchor_col
            
            # Find all words that can be placed starting at this position
            self.generate_moves(start_row, start_col, rack, direction, prefix_length)
    
    def calculate_prefix_limit(self, row, col, direction):
        """
        Calculate the maximum number of tiles that can be placed before an anchor point.
        
        Args:
            row, col: The anchor position
            direction: 'right' or 'down'
        
        Returns:
            Maximum number of tiles that can be placed before the anchor
        """
        limit = 0
        
        if direction == 'right':
            # Count empty squares to the left
            curr_col = col - 1
            while curr_col >= 0:
                node = self.get_node_at(row, curr_col)
                if node and not node.occupied:
                    limit += 1
                    curr_col -= 1
                else:
                    break
        else:  # direction == 'down'
            # Count empty squares above
            curr_row = row - 1
            while curr_row >= 0:
                node = self.get_node_at(curr_row, col)
                if node and not node.occupied:
                    limit += 1
                    curr_row -= 1
                else:
                    break
        
        return limit
    
    def generate_moves(self, start_row, start_col, rack, direction, prefix_length):
        """
        Generate all possible moves starting from a specific position.
        
        Args:
            start_row, start_col: Starting position
            rack: Available letters
            direction: 'right' or 'down'
            prefix_length: How far into the word is the anchor point
        """
        # Start with the root of the dictionary trie
        self.generate_moves_recursive("", self.dictionary.root, start_row, start_col, 
                                     rack, direction, prefix_length, [], False)
    
    # def generate_moves_recursive(self, partial_word, dict_node, row, col, available_rack, 
    #                             direction, remaining_prefix, placed_tiles, word_has_anchor):
    #     """
    #     Recursively generate all valid moves starting from a position.
        
    #     Args:
    #         partial_word: Word built so far
    #         dict_node: Current node in the dictionary trie
    #         row, col: Current position on the board
    #         available_rack: Letters still available in the rack
    #         direction: 'right' or 'down'
    #         remaining_prefix: How many more letters before reaching the anchor
    #         placed_tiles: List of tiles placed so far [(position, letter)]
    #         word_has_anchor: Whether the word uses an existing anchor point
    #     """
    #     # Check if we're still on the board
    #     node = self.get_node_at(row, col)
    #     if not node:
    #         # We've gone off the board, so check if we have a valid word
    #         if dict_node.is_terminal and word_has_anchor and partial_word:
    #             # We have a complete word that uses an anchor
    #             self.record_move(partial_word, placed_tiles, direction)
    #         return
        
    #     # If this square is already occupied, we must use that letter
    #     if node.occupied:
    #         # Get the letter on this square
    #         letter = node.tile
            
    #         # Check if this letter continues a valid path in our dictionary
    #         next_node = dict_node.get_child(letter)
    #         if next_node:
    #             # This letter is valid, so continue building the word
                
    #             # Calculate next position
    #             next_row, next_col = self.get_next_position(row, col, direction)
                
    #             # Continue recursively
    #             self.generate_moves_recursive(
    #                 partial_word + letter,
    #                 next_node,
    #                 next_row,
    #                 next_col,
    #                 available_rack,
    #                 direction,
    #                 0,  # No more prefix needed since we've already placed at least one tile
    #                 placed_tiles + [((row, col), None)],  # Mark that we used an existing tile
    #                 True  # We've now used at least one anchor
    #             )
    #     else:
    #         # The square is empty, we can place any letter from our rack
            
    #         # If we need to place a prefix tile, we don't check cross-constraints yet
    #         if remaining_prefix > 0:
    #             valid_letters = set(available_rack)
    #         else:
    #             # Get the set of valid letters based on cross-checks
    #             valid_letters = self.get_cross_checks(row, col, direction).intersection(available_rack)
            
    #         # Try each valid letter
    #         for letter in valid_letters:
    #             # Check if this letter continues a valid path in our dictionary
    #             next_node = dict_node.get_child(letter)
    #             if next_node:
    #                 # This letter is valid, so continue building the word
                    
    #                 # Remove the letter from the rack
    #                 rack_copy = available_rack.copy()
    #                 rack_copy.remove(letter)
                    
    #                 # Calculate next position
    #                 next_row, next_col = self.get_next_position(row, col, direction)
                    
    #                 # Continue recursively
    #                 self.generate_moves_recursive(
    #                     partial_word + letter,
    #                     next_node,
    #                     next_row,
    #                     next_col,
    #                     rack_copy,
    #                     direction,
    #                     max(0, remaining_prefix - 1),
    #                     placed_tiles + [((row, col), letter)],
    #                     word_has_anchor or remaining_prefix == 0  # A tile at the anchor counts
    #                 )
            
    #         # If we have a valid word so far and we've used an anchor, record it
    #         if dict_node.is_terminal and word_has_anchor and partial_word:
    #             self.record_move(partial_word, placed_tiles, direction)

    def generate_moves_recursive(self, partial_word, dict_node, row, col, available_rack, 
                                direction, remaining_prefix, placed_tiles, word_has_anchor):
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
        node = self.get_node_at(row, col)
        if not node:
            # We've gone off the board, so check if we have a valid word
            if dict_node.is_terminal and word_has_anchor and partial_word:
                # We have a complete word that uses an anchor
                self.record_move(partial_word, placed_tiles, direction)
            return
        
        # If this square is already occupied, we must use that letter
        if node.occupied:
            # Get the letter on this square
            letter = node.tile
            
            # Check if this letter continues a valid path in our dictionary
            next_node = dict_node.get_child(letter)
            if next_node:
                # This letter is valid, so continue building the word
                
                # Calculate next position
                next_row, next_col = self.get_next_position(row, col, direction)
                
                # Continue recursively
                self.generate_moves_recursive(
                    partial_word + letter,
                    next_node,
                    next_row,
                    next_col,
                    available_rack,
                    direction,
                    0,  # No more prefix needed since we've already placed at least one tile
                    placed_tiles + [((row, col), None)],  # Mark that we used an existing tile
                    True  # We've now used at least one anchor
                )
        else:
            # The square is empty, we can place any letter from our rack
            
            # If we need to place a prefix tile, we don't check cross-constraints yet
            if remaining_prefix > 0:
                valid_letters = set(available_rack)
            else:
                # Get the set of valid letters based on cross-checks
                valid_letters = self.get_cross_checks(row, col, direction).intersection(set(available_rack))
            
            # Try each valid letter
            for letter in valid_letters:
                # Check if this letter continues a valid path in our dictionary
                next_node = dict_node.get_child(letter)
                if next_node:
                    # This letter is valid, so continue building the word
                    
                    # Remove the letter from the rack
                    rack_copy = available_rack.copy()
                    rack_copy.remove(letter)
                    
                    # Calculate next position
                    next_row, next_col = self.get_next_position(row, col, direction)
                    
                    # Continue recursively
                    self.generate_moves_recursive(
                        partial_word + letter,
                        next_node,
                        next_row,
                        next_col,
                        rack_copy,
                        direction,
                        max(0, remaining_prefix - 1),
                        placed_tiles + [((row, col), letter)],
                        word_has_anchor or remaining_prefix == 0  # A tile at the anchor counts
                    )
            
            # If we have a valid word so far and we've used an anchor, record it
            if dict_node.is_terminal and word_has_anchor and partial_word:
                self.record_move(partial_word, placed_tiles, direction)

    def get_next_position(self, row, col, direction):
        """Get the next position based on the current direction."""
        if direction == 'right':
            return row, col + 1
        else:  # direction == 'down'
            return row + 1, col

    def is_anchor_square(self, row, col):
        """
        Check if a square is an anchor square (empty but adjacent to a tile).
        
        Args:
            row, col: Position to check
        
        Returns:
            Boolean indicating if this is an anchor square
        """
        node = self.get_node_at(row, col)
        if not node or node.occupied:
            return False
            
        # Check if this empty square is adjacent to any occupied square
        adjacent_positions = [
            (row-1, col),   # up
            (row+1, col),   # down
            (row, col-1),   # left
            (row, col+1)    # right
        ]
        
        for adj_row, adj_col in adjacent_positions:
            adj_node = self.get_node_at(adj_row, adj_col)
            if adj_node and adj_node.occupied:
                return True
                
        return False
    
    def record_move(self, word, placed_tiles, direction):
        """
        Record a valid move in the list of valid moves.
        
        Args:
            word: The word formed
            placed_tiles: List of tiles placed [(position, letter)]
            direction: 'right' or 'down'
        """
        # Calculate score for this placement
        self.direction = direction  # Needed for score calculation
        score = self.calculate_placement_score(placed_tiles)
        
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
        
        self.valid_moves.append(move)
    
    def calculate_placement_score(self, placed_tiles):
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
            node = self.get_node_at(row, col)
            
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
                cross_word_score = self.calculate_cross_word_score(row, col, letter, self.direction)
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
    
    def calculate_cross_word_score(self, row, col, letter, direction):
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
        cross_direction = 'down' if direction == 'right' else 'right'
        
        # Check if this letter forms a cross-word
        # A cross-word is formed if there are adjacent tiles in the perpendicular direction
        if cross_direction == 'down':
            # Check for tiles above or below
            if not (self.get_node_at(row-1, col) and self.get_node_at(row-1, col).occupied) and \
               not (self.get_node_at(row+1, col) and self.get_node_at(row+1, col).occupied):
                return 0  # No cross-word formed
        else:  # cross_direction == 'right'
            # Check for tiles to the left or right
            if not (self.get_node_at(row, col-1) and self.get_node_at(row, col-1).occupied) and \
               not (self.get_node_at(row, col+1) and self.get_node_at(row, col+1).occupied):
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
                upper_node = self.get_node_at(curr_row-1, col)
                if upper_node and upper_node.occupied:
                    curr_row -= 1
                else:
                    break
            
            # Build the word from top to bottom
            start_row = curr_row
            while True:
                curr_node = self.get_node_at(curr_row, col)
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
                elif curr_node.occupied:
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
                left_node = self.get_node_at(row, curr_col-1)
                if left_node and left_node.occupied:
                    curr_col -= 1
                else:
                    break
            
            # Build the word from left to right
            start_col = curr_col
            while True:
                curr_node = self.get_node_at(row, curr_col)
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
                elif curr_node.occupied:
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
    
    def get_best_move(self):
        """
        Get the best move based on the selected strategy.
        
        Returns:
            The best move according to the selected strategy
        """
        if self.strategy == "greedy":
            # Simple greedy strategy - just pick the highest scoring move
            return self.get_greedy_move()
        elif self.strategy == "mcts":
            # Monte Carlo Tree Search strategy
            return self.get_mcts_move()
        elif self.strategy == "beam":
            # Beam Search strategy
            return self.get_beam_search_move()
        else:
            # Default to greedy
            return self.get_greedy_move()
    
    def get_greedy_move(self):
        """
        Get the highest scoring move using a simple greedy approach.
        
        Returns:
            The highest scoring move
        """
        # Find all valid moves
        moves = self.find_all_moves()
        
        # Return the highest scoring move, or None if no moves are available
        if moves:
            return moves[0]  # Moves are already sorted by score
        return None
    
    def get_mcts_move(self):
        """
        Get the best move using Monte Carlo Tree Search.
        
        Returns:
            The best move according to MCTS
        """
        # Find all valid moves first
        moves = self.find_all_moves()
        if not moves:
            return None
            
        # Set up the initial state for MCTS
        initial_state = {
            'board': self.board,
            'player': {'score': 0},  # Simplified player object for simulation
            'legal_moves': moves
        }
        
        # Run MCTS
        best_move = monte_carlo_tree_search(initial_state)
        return best_move
    
    def get_beam_search_move(self):
        """
        Get the best move using Beam Search.
        
        Returns:
            The best move according to Beam Search
        """
        # Create a beam search instance
        beam_search = BeamSearchScrabble(self.rack, self.board)
        
        # Find the best move
        best_move = beam_search.find_best_move()
        return best_move

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