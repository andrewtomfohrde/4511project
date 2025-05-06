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