from itertools import permutations

def load_dictionary(file_path="dic.txt"):
    """Load dictionary from file and return as a set of uppercase words."""
    with open(file_path, 'r') as f:
        words = set(line.strip().upper() for line in f)
    return words

def get_possible_words(board, rack, dictionary, player):
    """
    Generate legal horizontal words that can be placed on the board using rack letters.
    Scores are calculated using real board logic.
    """
    legal_moves = get_possible_words(board, player.rack, dictionary, player)
    board_size = 15
    anchor_points = []

    board_data = board.board_array()

    # Step 1: Find anchor points (existing letters)
    for row in range(board_size):
        for col in range(board_size):
            if board_data[row][col] != "   " and board_data[row][col] not in ["TLS", "DLS", "TWS", "DWS", " * "]:
                anchor_points.append((row, col))

    # Step 2: Fallback to center if empty board
    if not anchor_points:
        anchor_points = [(7, 7)]

    # Step 3: Try horizontal placements at anchor points
    for row, col in anchor_points:
        for word in dictionary:
            if len(word) > 7:
                continue  # Skip unplayable words

            if col + len(word) > 15:
                continue  # Would overflow board

            temp_word = Word(word, [row, col], player, "right", board_data)
            if temp_word.check_word() is True:
                player.score = 0  # Reset mock scoring
                temp_word.calculate_word_score()
                score = player.get_score()
                player.score = 0

                move = {
                    'word': word,
                    'score': score,
                    'position': (row, col),
                    'direction': 'right'
                }
                legal_moves.append(move)

    return legal_moves

def can_form_word(word, rack, board_data, row, col):
    """
    Check if the word can be placed at (row, col) going horizontally using the player's rack.
    Must use existing letters on board when applicable.
    """
    if col + len(word) > 15:
        return False  # Word doesn’t fit

    rack_copy = list(rack.get_rack_str())  # Convert rack string to list of characters
    placed = False

    for i in range(len(word)):
        board_tile = board_data[row][col + i]
        if board_tile == "   " or board_tile in ["TLS", "DLS", "TWS", "DWS", " * "]:
            if word[i] in rack_copy:
                rack_copy.remove(word[i])
            elif '_' in rack_copy:
                rack_copy.remove('_')  # Use blank tile if available
            else:
                return False
        elif board_tile[1] == word[i]:  # Match existing tile letter (middle char of " A ")
            placed = True
        else:
            return False  # Conflict with existing tile

    return placed or True  # Allow if connecting or first move
