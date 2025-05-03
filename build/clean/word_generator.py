from itertools import permutations
# from scrabble import Word


def load_dictionary(filename="scrabbledict.txt"):
    with open(filename) as f:
        return f.read().splitlines()

def get_possible_words(board, rack, dictionary, player, Word):
    legal_moves = []
    board_size = 15
    anchor_points = []

    # Step 1: Identify anchor points (occupied tiles)
    for row in range(board_size):
        for col in range(board_size):
            node = board.get_node(row, col)
            if node.occupied:
                anchor_points.append((row, col))

    if not anchor_points:
        anchor_points = [(7, 7)]

    # Step 2: Convert rack to multiset
    from collections import Counter
    rack_letters = Counter([tile.get_letter() for tile in rack.get_rack_arr()])

    # Step 3: Try horizontal placements at anchor points
    for row, col in anchor_points:
        for word in dictionary:
            if len(word) > 7 or col + len(word) > 15:
                continue

            # Estimate which tiles would need to be used from the rack
            letters_needed = list(word)
            for i in range(len(word)):
                board_node = board.get_node(row, col + i)
                if board_node and board_node.occupied and board_node.char == word[i]:
                    letters_needed[i] = None  # already on board

            letters_to_find = [ltr for ltr in letters_needed if ltr is not None]
            available = rack_letters.copy()

            can_make = True
            for ltr in letters_to_find:
                if available[ltr] > 0:
                    available[ltr] -= 1
                elif available["#"] > 0:  # try using a blank
                    available["#"] -= 1
                else:
                    can_make = False
                    break

            if not can_make:
                continue

            # Try placement
            test_word = Word(word, [row, col], player, "right", board)
            valid, placed = test_word.check_word()

            if valid:
                score = test_word.calculate_word_score(placed)
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
