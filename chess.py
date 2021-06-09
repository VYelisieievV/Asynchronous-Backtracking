import numpy as np
from utils import plot_chess
import matplotlib.pyplot as plt


def knight_rule_attack(x, y, x_ref, y_ref, n):
    """
    Function to constraint a knight.
    Checks if a knight figure can perform an attack move on target.

    Args:
        x (int): x coordinate of knight
        y (int): y coordinate of knight
        x_ref (int): x coordinate for target
        y_ref (int): y coordinate for target
        n (int): field size, unused

    Returns:
        bool: True if attack is possible, False otherwise
    """
    possible_coords = [
        (x + 2, y + 1),
        (x + 2, y - 1),
        (x - 2, y + 1),
        (x - 2, y - 1),
        (x, y),
        (x + 1, y + 2),
        (x + 1, y - 2),
        (x - 1, y + 2),
        (x - 1, y - 2),
    ]
    for x_pos, y_pos in possible_coords:
        if x_pos == x_ref and y_pos == y_ref:
            return True
    return False


def king_rule_attack(x, y, x_ref, y_ref, n):
    """
    Function to constraint a king.
    Checks if a king figure can perform an attack move on target.

    Args:
        x (int): x coordinate of king
        y (int): y coordinate of king
        x_ref (int): x coordinate for target
        y_ref (int): y coordinate for target
        n (int): field size, unused

    Returns:
        bool: True if attack is possible, False otherwise
    """

    possible_coords = [
        (x + 1, y + 1),
        (x + 1, y),
        (x + 1, y - 1),
        (x - 1, y + 1),
        (x, y + 1),
        (x, y - 1),
        (x - 1, y - 1),
        (x - 1, y),
        (x, y),
    ]
    for x_pos, y_pos in possible_coords:
        if x_pos == x_ref and y_pos == y_ref:
            return True
    return False


def bbishop_rule_attack(x, y, x_ref, y_ref, n):
    """
    Function to constraint a black bishop. Black bishop is the one which can move only on black squares.
    Checks if a black bishop figure stands on correct square and can perform an attack move on target.

    Args:
        x (int): x coordinate of black bishop
        y (int): y coordinate of black bishop
        x_ref (int): x coordinate for target
        y_ref (int): y coordinate for target
        n (int): field size, unused

    Returns:
        bool: True if satisfies constrain (correct position and can attack), False otherwise
    """
    is_white = (np.indices((x + 1, y + 1)).sum(axis=0) % 2)[-1, -1] == 1
    if is_white:
        return True

    possible_coords = (
        [(x, y)]
        + list(zip(range(x + 1, n), range(y + 1, n)))
        + list(zip(range(x + 1, n), range(y - 1, -1, -1)))
        + list(zip(range(x - 1, -1, -1), range(y + 1, n)))
        + list(zip(range(x - 1, -1, -1), range(y - 1, -1, -1)))
    )

    for x_pos, y_pos in possible_coords:
        if x_pos == x_ref and y_pos == y_ref:
            return True
    return False


def wbishop_rule_attack(x, y, x_ref, y_ref, n):
    """
    Function to constraint a white bishop. White bishop is the one which can move only on white squares.
    Checks if a white bishop figure stands on correct square and can perform an attack move on target.

    Args:
        x (int): x coordinate of white bishop
        y (int): y coordinate of white bishop
        x_ref (int): x coordinate for target
        y_ref (int): y coordinate for target
        n (int): field size, unused

    Returns:
        bool: True if satisfies constrain (correct position and can attack), False otherwise
    """
    is_black = (np.indices((x + 1, y + 1)).sum(axis=0) % 2)[-1, -1] == 0
    if is_black:
        return True
    possible_coords = (
        [(x, y)]
        + list(zip(range(x + 1, n), range(y + 1, n)))
        + list(zip(range(x + 1, n), range(y - 1, -1, -1)))
        + list(zip(range(x - 1, -1, -1), range(y + 1, n)))
        + list(zip(range(x - 1, -1, -1), range(y - 1, -1, -1)))
    )
    for x_pos, y_pos in possible_coords:
        if x_pos == x_ref and y_pos == y_ref:
            return True
    return False


class Chessboard(object):
    def __init__(
        self,
        size: int,
        figure_config,
    ):
        self.desk = np.zeros((size, size))
        self.size = size
        self.figures = []
        self.figure_config = dict()
        for el in figure_config:
            self.figure_config[el[0]] = {
                "color": el[2],
                "n": el[1],
                "number": el[3],
                "const_attack": el[4],
            }
            if el[0] == "knight":
                ind = np.random.choice(self.size ** 2, el[1], replace=False)
                indx, indy = ind // self.size, ind % self.size
                for i in range(len(ind)):
                    self.desk[indx[i], indy[i]] = el[3]
                self.figures += [
                    [np.array([indx[i], indy[i]]), el[0]] for i in range(len(ind))
                ]

            elif el[0] == "wbishop":
                chess_pattern = np.indices((self.size, self.size)).sum(axis=0) % 2  #
                # generate all white
                taken = np.array([first[0] for first in self.figures])
                indices = np.argwhere((chess_pattern == 1))
                if taken.size != 0:
                    indices = indices[
                        np.all(np.any((indices - taken[:, None]), axis=2), axis=0)
                    ]
                indices = indices[
                    np.random.choice(indices.shape[0], el[1], replace=False)
                ]

                for indx in indices:
                    self.desk[indx[0], indx[1]] = el[3]
                self.figures += [[indx, el[0]] for indx in indices]

            elif el[0] == "bbishop":
                chess_pattern = np.indices((self.size, self.size)).sum(axis=0) % 2  #
                # generate all black
                taken = np.array([first[0] for first in self.figures])
                indices = np.argwhere((chess_pattern == 0))
                if taken.size != 0:
                    indices = indices[
                        np.all(np.any((indices - taken[:, None]), axis=2), axis=0)
                    ]
                indices = indices[
                    np.random.choice(indices.shape[0], el[1], replace=False)
                ]

                for indx in indices:
                    self.desk[indx[0], indx[1]] = el[3]
                self.figures += [[indx, el[0]] for indx in indices]

    def show_field(self):
        field = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for figure in self.figures:
            coord, name = figure
            field[coord[0], coord[1]] = self.figure_config[name]["color"]

        plot_chess(field)
        plt.show()

    def zero_field(self):
        self.desk = np.zeros((self.size, self.size))
        self.figure = [(None, el[1]) for el in self.figures]
