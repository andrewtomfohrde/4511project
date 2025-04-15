# mcts_scrabble.py

import random
import copy
from word_generator import load_dictionary, get_possible_words
from scrabble import Word



class Node:
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
    word_obj = Word(move['word'], list(move['position']), temp_player, move['direction'], temp_board.board_array())
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
    child = Node(new_state, parent=node, move=move)
    node.children.append(child)
    return child

def backpropagate(node, result):
    while node:
        node.visits += 1
        node.total_score += result
        node = node.parent

def monte_carlo_tree_search(initial_state, iterations=1000):
    root = Node(initial_state)

    for _ in range(iterations):
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        leaf = expand(node)
        result = rollout(leaf.state)
        backpropagate(leaf, result)

    best = max(root.children, key=lambda c: c.visits)
    return best.move

if __name__ == "__main__":
    from word_generator import load_dictionary, get_possible_words

    # Example placeholder objects for standalone testing
    class FakeBoard:
        def __init__(self):
            self.grid = [[" " for _ in range(15)] for _ in range(15)]

    class FakePlayer:
        def __init__(self, rack):
            self.rack = rack

    board = FakeBoard()
    player = FakePlayer(['S', 'T', 'A', 'R', 'E', 'L', 'D'])

    dictionary = load_dictionary("dic.txt")
    legal_moves = get_possible_words(board, player.rack, dictionary)

    initial_state = {
        'board': board,
        'rack': player.rack,
        'legal_moves': legal_moves,
        'player': player  # ← Add this
    }

    best_move = monte_carlo_tree_search(initial_state, iterations=500)
    print("Best Move Found:", best_move)
