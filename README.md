# sudoku_AC3_solver
 
This is an advanced Sudoku Solver using Improved AC-3 Algorithm + MRV Heuristic

Tradinitionally, sudoku solver was implemented using simple DFS algorithms, which can solve some easy sudoku problems but failed when the problem gets more complicated. Here I used an improved AC-3 algorithm with MRV heurisitc to solve pretty much all the sudoku problems out there, including some of the hardest, fast and efficiently.

How to use the GUI to visualize:

1. In sudokuGUI, go to line 276, where it says:
game = SudokuGame('090700860031005020806000000007050006000307000500010700000000109020600350054008070')

Pick the sudoku board that you want to play (You can go the sudoku file to pick one or import your favorite sudoku board using the correct format.)

2. Tn terminal, run:
python3 sudokuGUI.py

3. Then a UI will pop out, depends on the level of the puzzle, you can choose one of three versions of AC-3 solvers. For the hard version, only "infer-with-guessing" can solve it.
Generally,  infer_ac3 < infer_improved < infer_with_guessing