from random import shuffle
import re

# === Letter Scores ===
LETTER_VALUES = {
    "A": 1, "B": 3, "C": 3, "D": 2, "E": 1, "F": 4, "G": 2, "H": 4, "I": 1,
    "J": 8, "K": 5, "L": 1, "M": 3, "N": 1, "O": 1, "P": 3, "Q": 10, "R": 1,
    "S": 1, "T": 1, "U": 1, "V": 4, "W": 4, "X": 8, "Y": 4, "Z": 10, "#": 0
}

# === Tile ===
class Tile:
    def __init__(self, letter):
        self.letter = letter.upper()
        self.score = LETTER_VALUES.get(self.letter, 0)

    def get_letter(self):
        return self.letter

    def get_score(self):
        return self.score

# === Bag ===
class Bag:
    def __init__(self):
        self.bag = []
        self.initialize_bag()

    def add_to_bag(self, letter, quantity):
        for _ in range(quantity):
            self.bag.append(Tile(letter))

    def initialize_bag(self):
        tile_counts = {
            "A": 9, "B": 2, "C": 2, "D": 4, "E": 12, "F": 2, "G": 3, "H": 2,
            "I": 9, "J": 1, "K": 1, "L": 4, "M": 2, "N": 6, "O": 8, "P": 2,
            "Q": 1, "R": 6, "S": 4, "T": 6, "U": 4, "V": 2, "W": 2, "X": 1,
            "Y": 2, "Z": 1, "#": 2
        }
        for letter, count in tile_counts.items():
            self.add_to_bag(letter, count)
        shuffle(self.bag)

    def take_from_bag(self):
        return self.bag.pop() if self.bag else None

    def get_remaining_tiles(self):
        return len(self.bag)

# === Rack ===
class Rack:
    def __init__(self, bag):
        self.rack = []
        self.bag = bag
        self.initialize()

    def add_to_rack(self):
        tile = self.bag.take_from_bag()
        if tile:
            self.rack.append(tile)

    def initialize(self):
        while len(self.rack) < 7:
            self.add_to_rack()

    def get_rack_str(self):
        return ", ".join(tile.get_letter() for tile in self.rack)

    def get_rack_arr(self):
        return self.rack

    def remove_from_rack(self, tile):
        self.rack.remove(tile)

    def get_rack_length(self):
        return len(self.rack)

    def replenish_rack(self):
        while len(self.rack) < 7 and self.bag.get_remaining_tiles() > 0:
            self.add_to_rack()

# === Player ===
class Player:
    def __init__(self, bag):
        self.name = ""
        self.rack = Rack(bag)
        self.score = 0

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_rack_str(self):
        return self.rack.get_rack_str()

    def get_rack_arr(self):
        return self.rack.get_rack_arr()

    def increase_score(self, increase):
        self.score += increase

    def get_score(self):
        return self.score

# === BoardNode ===
class BoardNode:
    def __init__(self):
        self.tile = None
        self.char = ' '
        self.occupied = False
        self.position = (0, 0)
        self.score_multiplier = ""
        self.right = self.left = self.up = self.down = None

    def place_tile(self, tile):
        if self.occupied:
            return False
        self.tile = tile
        self.char = tile.get_letter()
        self.occupied = True
        return True

    def place_blank(self, tile, char):
        if self.occupied:
            return False
        self.tile = tile
        self.char = char
        self.occupied = True
        return True

    def get_display_str(self):
        if self.occupied:
            return f" {self.char} "
        elif self.position == (7, 7):
            return " * "
        elif self.score_multiplier:
            return self.score_multiplier
        else:
            return "   "

# === ScrabbleBoard ===
class ScrabbleBoard:
    def __init__(self):
        self.size = 15
        self.nodes = [[BoardNode() for _ in range(15)] for _ in range(15)]
        for i in range(15):
            for j in range(15):
                self.nodes[i][j].position = (i, j)
        for i in range(15):
            for j in range(14):
                self.nodes[i][j].right = self.nodes[i][j + 1]
                self.nodes[i][j + 1].left = self.nodes[i][j]
        for i in range(14):
            for j in range(15):
                self.nodes[i][j].down = self.nodes[i + 1][j]
                self.nodes[i + 1][j].up = self.nodes[i][j]

    def get_node(self, row, col):
        if 0 <= row < 15 and 0 <= col < 15:
            return self.nodes[row][col]
        return None

    def get_board_str(self):
        result = ""
        for i in range(15):
            row = [self.nodes[i][j].get_display_str() for j in range(15)]
            result += f"{i:2} | " + " | ".join(row) + " |\n"
        return result

# === Word class ===
class Word:
    def __init__(self, word, location, player, direction, board):
        self.word = word.upper()
        self.location = location
        self.player = player
        self.direction = direction
        self.board = board
