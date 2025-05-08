def get_dfs_move(board, rack, dict):
    valid_moves = find_all_moves(board, rack, dict)
    move_tree = create_word_tree(valid_moves, rack)

    print(move_tree)
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
            print(f"{child_node.word} set at {child_node.position} wit score {child_node.score}!")
            child_score = child_node.score
            if child_score > best_score:
                best_score = child_score
                best_move = child_result
    
    return best_move
