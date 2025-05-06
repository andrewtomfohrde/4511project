    
def get_bfs_move(board, rack, dict):
    valid_moves = find_all_moves(board, rack, dict)
    best_move = None

    if valid_moves:
        word, pos, dir, placed, score = valid_moves[0]
        best_move = [word, pos, dir]

    return best_move