import copy
from typing import List, Tuple, Dict, Optional
from scrabble import ScrabbleBoard, Game
from player import Bag, Tile, Rack, Player
import algos
from word import Word
from algos import ScrabbleAI

class Simulation(Game):
    def __init__(self, dict):
        
        self.board = None
        self.players = []
        self.bag = None
        self.dict = dict
    
    def deep_copy_board(board):
        new_board = ScrabbleBoard()
        # Map from original nodes to new nodes for reference
        node_map = {}
        # First pass: Create all nodes with their properties (except connections)
        original_node = board.start_node
        row = 0
        while original_node:
            col_node = original_node
            col = 0
            while col_node:
                # Find the corresponding new node
                new_node = Simulation._get_node_at_position(new_board.start_node, (row, col))
                
                # Copy properties
                if col_node.tile:
                    new_node.tile = copy.deepcopy(col_node.tile)
                new_node.score_multiplier = col_node.score_multiplier
                
                # Add to map
                node_map[col_node] = new_node
                
                # Move to next column
                col_node = col_node.right
                col += 1
                
            # Move to next row
            original_node = original_node.down
            row += 1
            
        return new_board
    
    def _get_node_at_position(start_node, position):
        """
        Get the node at a specific position starting from the top-left node.
        """
        row, col = position
        current = start_node
        
        # Move down to the correct row
        for _ in range(row):
            if current.down:
                current = current.down
            else:
                return None
        
        # Move right to the correct column
        for _ in range(col):
            if current.right:
                current = current.right
            else:
                return None
        
        return current
    
    def deep_copy_bag(bag):
        """
        Create a deep copy of the Bag object.
        """
        return copy.deepcopy(bag)
    
    def deep_copy_player(self, player, new_board, new_bag):
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
                player.dict,  # Dictionary can be shared as it doesn't change
                new_board,
                new_bag,
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
    
    def deep_copy_game_state(self, board, bag, players, dict):
        """
        Create a deep copy of the entire game state.
        
        Args:
            game_state: A dictionary containing 'board', 'bag', 'current_player', and 'players'
            
        Returns:
            A deep copy of the game state
        """
        # Copy the board
        new_board = Simulation.deep_copy_board(board)
        self.board = new_board
        # Copy the bag
        new_bag = Simulation.deep_copy_bag(bag)
        self.bag = new_bag
        # Copy all players
        new_players = []
        for player in players:
            new_player = Simulation.deep_copy_player(player, new_board, new_bag)
            new_players.append(new_player)
        self.players = new_players
    
        # Create the new game state
        new_game_state = {
            'board': new_board,
            'bag': new_bag,
            'players': new_players,
            'dict': dict
        }
        
        return new_game_state
    
def get_mcts_move(board, rack, dictionary, num_simulations=1000, exploration_weight=1.0):
    """
    Use Monte Carlo Tree Search to find the best move.
    
    Args:
        board: The current ScrabbleBoard
        rack: The current player's rack
        dictionary: The DictionaryTrie object
        num_simulations: Number of simulations to run
        exploration_weight: Controls exploration vs exploitation
        
    Returns:
        The best move found
    """
    # First, generate candidate moves
    candidate_moves = generate_candidate_moves(board, rack, dictionary)
    
    if not candidate_moves:
        return None  # No valid moves
    
    # Create game state
    game_state = {
        'board': board,
        'bag': get_current_bag(),  # You'll need to implement this
        'players': get_players(),  # You'll need to implement this
        'current_player': get_current_player()  # You'll need to implement this
    }
    
    # Dictionary to store statistics for each move
    move_stats = {move: {'wins': 0, 'plays': 0} for move in candidate_moves}
    
    # Run simulations
    for _ in range(num_simulations):
        # Select a move to explore
        # Use UCB1 formula for selection
        best_move = None
        best_value = float('-inf')
        
        for move in candidate_moves:
            stats = move_stats[move]
            
            # If move hasn't been played yet, prioritize it
            if stats['plays'] == 0:
                best_move = move
                break
                
            # Calculate UCB1 value
            exploitation = stats['wins'] / stats['plays']
            exploration = exploration_weight * (2 * (total_plays := sum(s['plays'] for s in move_stats.values())) / stats['plays'])
            ucb_value = exploitation + exploration
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_move = move
        
        # Simulate the game after making this move
        # Create a deep copy of the game state
        sim_game_state = Simulation.deep_copy_game_state(game_state)
        
        # Make the move in the simulation
        make_move_in_simulation(sim_game_state, best_move)
        
        # Play the game to completion and get the result
        result = simulate_to_end(sim_game_state)
        
        # Update statistics
        stats = move_stats[best_move]
        stats['plays'] += 1
        if result > 0:  # If the current player won
            stats['wins'] += 1
    
    # Choose the move with the best win rate
    best_move = max(candidate_moves, key=lambda move: move_stats[move]['wins'] / move_stats[move]['plays'] if move_stats[move]['plays'] > 0 else 0)
    
    return best_move