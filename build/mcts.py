def get_mcts_move(board, rack, dict):
    import random

    valid_moves = find_all_moves(board, rack, dict)
    best_move = None

    if not valid_moves:
        return None

    # MCTS heuristic: sample moves randomly and pick the one with the highest score
    best_score = -1
    simulations = min(50, len(valid_moves))  # run up to 50 random simulations

    for _ in range(simulations):
        move = random.choice(valid_moves)
        word, pos, dir, placed, score = move
        if score > best_score:
            best_score = score
            best_move = [word, pos, dir]

    return best_move
