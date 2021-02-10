############################################################
# CIS 521: Homework 4
############################################################

student_name = "Yuxuan Hong"

############################################################
# Imports
############################################################
import copy
from collections import deque
from queue import PriorityQueue
import heapq



############################################################
# Section 1: Sudoku Solver
############################################################

def sudoku_cells():
    cells = []
    for i in range(9):
        for j in range(9):
            cells.append((i,j))
    return cells

def sudoku_arcs():
    arcs = []
    cells = sudoku_cells()
    for cell_1 in cells:
        for cell_2 in cells:
            if cell_1 == cell_2:
                continue
            # same row
            if cell_1[0] == cell_2[0]:
                arcs.append((cell_1, cell_2))
            
            # same col
            if cell_1[1] == cell_2[1]:
                arcs.append((cell_1, cell_2))

            # same block
            if cell_1[0] // 3 == cell_2[0] // 3 and cell_1[1] // 3 == cell_2[1] // 3 and (cell_1, cell_2) not in arcs:
                arcs.append((cell_1, cell_2))
    # arcs = set(arcs)
    return arcs

def read_board(path):
    f = open(path, 'r')
    content = f.readlines()
    bd_array = [list(x.strip()) for x in content]

    board = {}
    for i in range(len(bd_array)):
        for j in range(len(bd_array[0])):
            if bd_array[i][j] != '*':
                board[(i,j)] = set([int(bd_array[i][j])])
            else:
                board[(i,j)] = set(range(1, len(bd_array) + 1))
    # print(board)
    return board

class Sudoku(object):

    CELLS = sudoku_cells()
    ARCS = sudoku_arcs()

    def __init__(self, board):
        '''board repr: {key=cell, val=set(of values domain)}'''
        self.board = board

    def get_values(self, cell):
        '''return the set of vals currently available at a cell'''
        return self.board[cell]

    def remove_inconsistent_values(self, cell1, cell2):
        '''
        remove any vals in set for C1 for which there are no vals in the set for C2 that satisfy the inequality constraint. In other words, remove a val from C1 if:
        1- C1 & C2 form an ARC pair
        2- C2 has only one elem
        3- C2 is a subset of C1
        '''
        removed = False

        if (cell1, cell2) in self.ARCS:
            if self.is_cell_solved(cell2) and self.board[cell2].issubset(self.board[cell1]):
                self.board[cell1].difference_update(self.board[cell2])
                removed = True
        return removed

    def is_cell_solved(self, cell):
        ''' check if a given cell is solved'''
        if len(self.board[cell]) == 1:
            return True
        return False


    def infer_ac3(self):
        '''basic constraint check for all arcs'''
        q = deque()
        for arc in self.ARCS:
            if not (self.is_cell_solved(arc[0]) and self.is_cell_solved(arc[1])):
                q.append(arc)
        
        while q:
            arc = q.popleft()
            if self.remove_inconsistent_values(arc[0], arc[1]):
                # add back all neighbors
                for neighbor_arc in self.find_neighbors(arc[0], arc[1]):
                    q.append(neighbor_arc)

        return self


    def find_neighbors(self, cell1, cell2):
        '''find all neighbors to cell1, excluding cell2, in the form of (cell_x, cell1) where cell_x != cell2
        neighbors is a list of arcs '''
        
        neighbors = []
        for arc in self.ARCS:
            if arc[1] == cell1 and arc[0] != cell2:
                neighbors.append(arc)
        return neighbors


    def is_val_unique(self, cell, val, checking_direction):
        '''check if a val of a cell is unique at the current board assignment for a given area (row/col/block) '''
        val_unique = True

        if checking_direction == "row":
            for j in range(9):
                neighbor_cell = (cell[0], j)
                if neighbor_cell != cell and val in self.board[neighbor_cell]:
                    val_unique = False

        if checking_direction == "col":
            for i in range(9):
                neighbor_cell = (i, cell[1])
                if neighbor_cell != cell and val in self.board[neighbor_cell]:
                    val_unique = False

        if checking_direction == "block":
            block_init_row = cell[0] // 3 * 3
            block_init_col = cell[1] // 3 * 3
            for i in range(3):
                for j in range(3):
                    neighbor_cell = (block_init_row + i, block_init_col + j)
                    if neighbor_cell != cell and val in self.board[neighbor_cell]:
                        val_unique = False

        return val_unique
    
    def infer_improved_wrapper(self):
        '''loop through all cells, for each cell, check if any val inside the domain is unique across
        its rows, cols, and blocks. If it's unique, is_unique = T, update the domain of the cell to be
        that single val. Else False, break to next cell'''
        domain_changed = False

        if not self.is_consistent():
            return domain_changed

        for cell in self.CELLS:
            if len(self.board[cell]) > 1:
                for val in self.board[cell]:
                    if self.is_val_unique(cell, val, "row"):
                        self.board[cell] = set([val])
                        domain_changed = True
                        break
                    
                    if self.is_val_unique(cell, val, "col"):
                        self.board[cell] = set([val])
                        domain_changed = True
                        break

                    if self.is_val_unique(cell, val, "block"):
                        self.board[cell] = set([val])
                        domain_changed = True
                        break

        return domain_changed

    def infer_improved(self):
        '''run a AC-3 first, then continuousely checking for unique val in each cell, while find unique val, 
        update the cells with unique val (they become solved), then run AC-3 again.
        keep inferencing till there's no unique val for each cell'''
        self.infer_ac3()
        while self.infer_improved_wrapper():
            self.infer_ac3()
        
        return self

    def is_solved(self):
        '''check if the current borad is solved'''
        for cell in self.CELLS:
            if len(self.board[cell]) != 1:
                return False
        return True

    def is_consistent(self):
        '''check if current board assignment is consistent or not'''
        for cell in self.CELLS:
            if len(self.board[cell]) == 0:
                return False
        return True

    def MRV_heuristic(self):
        '''select the unsolved cell w/h the minimum remaining value (min len(domain))
        create a cell_heap that order them by len(remaining vals in a cell)
        heap = (len(domain), cell)
        '''
        H = []
        for cell, val_domain in self.board.items():
            if len(val_domain) > 1:
                H.append((len(val_domain), cell))
        
        heapq.heapify(H)
        mrv_cell = heapq.heappop(H)[1]
        return mrv_cell

    def backtrack_search(self):
        if self.is_solved():
            return self.board

        cell = self.MRV_heuristic()
        for val in self.board[cell]:
            new_sudoku = copy.deepcopy(self)
            new_sudoku.board[cell] = set([val])
            new_sudoku.infer_improved()
            
            if new_sudoku.is_consistent(): #still solvable after guessing
                results = new_sudoku.backtrack_search()
                if results:
                    return results
    
    def infer_with_guessing(self):
        self.infer_improved()
        self.board = self.backtrack_search()

############################################################
# Section 2: Dominoes Games
############################################################

def create_dominoes_game(rows, cols):
    bd=[ [False for j in range(cols)] for i in range(rows)]
    return DominoesGame(bd)


class DominoesGame(object):

    def __init__(self, board):
        self.board = board
        self.row = len(board)
        self.col = len(board[0])

    def get_board(self):
        return self.board

    def reset(self):
        for i in range(self.row):
            for j in range(self.col):
                if self.board[i][j] != False:
                    self.board[i][j] = False

    def is_legal_move(self, row, col, vertical):
        cover_range = [(row, col)]
        if vertical:
            cover_range.append((row+1, col))
        else:
            cover_range.append((row, col+1))
        assert(len(cover_range) == 2)

        for tile in cover_range:
            if tile[0] >= self.row or tile[0] < 0 or tile[1] >= self.col or tile[1] < 0:
                return False
            elif self.board[tile[0]][tile[1]] == True:
                return False
        return True
 
    def legal_moves(self, vertical):
        '''return a list of tuples that contains all legal moves for cur player | a move is a tuple (i, j)'''
        all_legal_moves = []

        for row in range(self.row):
            for col in range(self.col):
                if self.is_legal_move(row, col, vertical):
                    all_legal_moves.append((row,col))

        return all_legal_moves

    def perform_move(self, row, col, vertical):
        if self.is_legal_move(row, col, vertical):
            self.board[row][col] = True

            if vertical:
                self.board[row + 1][col] = True
            else:
                self.board[row][col + 1] = True

    def game_over(self, vertical):
        if len(self.legal_moves(vertical)) == 0:
            return True
        return False

    def copy(self):
        new_dmGame = copy.deepcopy(self)
        return new_dmGame

    def successors(self, vertical):
        '''return a lst of all successors
        a successor is a tuple (move, new_game after perfoming the move)
        '''

        all_successors = []
        legal_moves = self.legal_moves(vertical)

        for move in legal_moves:
            new_Dominoes = self.copy()
            new_Dominoes.perform_move(move[0], move[1], vertical)
            all_successors.append((move, new_Dominoes))
        
        return all_successors

    def get_random_move(self, vertical):
        pass

    def get_value(self, vertical):
        '''calculate the value at the leave node
        assume the game state passdown here has already been changed'''
        
        value = len(self.legal_moves(vertical)) - len(self.legal_moves(not vertical))
        return value
    
    def max_value(self, alpha, beta, limit, cur_depth, vertical, move, leaves_visited):
        '''return value, move'''
        if cur_depth == limit:
            leaves_visited += 1
            # print((self.get_value(vertical), move, leaves_visited))
            return (self.get_value(vertical), move, leaves_visited)
        
        v = float('-inf')
        for successor in self.successors(vertical):
            cur_depth += 1
            if cur_depth == 1:
                move = successor[0]
            suc_game = successor[1]

            v2, move2, leaves_visited = suc_game.min_value(alpha, beta, limit, cur_depth, not vertical, move, leaves_visited)

            if v2 > v:
                v, best_move = v2, move2
                alpha = max(alpha, v)
            
            if v >= beta:
                return (v, move, leaves_visited)
            
            cur_depth -= 1

        return (v, best_move, leaves_visited)

    def min_value(self, alpha, beta, limit, cur_depth, vertical, move, leaves_visited):
        '''return value, move'''
        if cur_depth == limit:
            leaves_visited += 1
            # print((self.get_value(vertical), move, leaves_visited))
            return (self.get_value(not vertical), move, leaves_visited)
        
        v = float('inf')
        for successor in self.successors(vertical):
            cur_depth += 1
            if cur_depth == 1:
                move = successor[0]
            suc_game = successor[1]

            v2, move2, leaves_visited = suc_game.max_value(alpha, beta, limit, cur_depth, not vertical, move, leaves_visited)

            if v2 < v:
                v, best_move = v2, move2
                beta = min(beta, v)
            
            if v <= alpha:
                return (v, move, leaves_visited)
            
            cur_depth -= 1

        return (v, best_move, leaves_visited)
        
    def get_best_move(self, vertical, limit):
        '''return 3 elem-tuple  ( best_move as(row, col),  its associated_val, num of leave nodes visied during the search  )
        board_val = num_moves_possible for current player - num_moves_possible for opponent
        implemented as alpha-beta search
        '''
        val, best_move, leaves_visited = self.max_value(float('-inf'), float('inf'), limit, 0, vertical, None, 0)
        
        return (best_move, val, leaves_visited)
        
############################################################
# Section 3: Feedback
############################################################

# Just an approximation is fine.
feedback_question_1 = 20

feedback_question_2 = """
xxx
"""

feedback_question_3 = """
xxx
"""
