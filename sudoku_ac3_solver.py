
'''
Sudoku Solver using Improved AC-3 Algorithm + MRV Heuristic

Created by Yuxuan Andy Hong
Oct 2020
'''

############################################################
# Imports
############################################################
import copy
from collections import deque
from queue import PriorityQueue
import heapq



############################################################
# Sudoku Solver - AC 3 + MRV Heuristic
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

