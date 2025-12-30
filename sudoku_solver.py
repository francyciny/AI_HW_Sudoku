import time
import heapq
import math
import itertools
from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Dict, Any, Optional

# Try to import PySAT for Task 2.2. If not present, the code will handle it gracefully.
try:
    from pysat.solvers import Glucose3
    from pysat.formula import CNF
    PYSAT_AVAILABLE = True
except ImportError:
    PYSAT_AVAILABLE = False

# ==========================================
# PART 1: GENERALIZED PROBLEM MODELING
# ==========================================

class SearchProblem(ABC):
    """
    Abstract base class for any search problem (Task 1: Generalization).
    """
    @abstractmethod
    def get_initial_state(self):
        pass

    @abstractmethod
    def is_goal(self, state) -> bool:
        pass

    @abstractmethod
    def get_successors(self, state) -> List[Tuple[Any, Any, float]]:
        """Returns list of (next_state, action, cost)"""
        pass

    @abstractmethod
    def get_heuristic(self, state) -> float:
        pass

class SudokuState:
    """
    Immutable representation of a Sudoku grid for hashing in A*.
    """
    def __init__(self, grid: Tuple[int], n: int):
        self.grid = grid
        self.n = n  # Size of the grid (e.g., 9 for 9x9)
        self.size = int(math.sqrt(n)) # Box size (e.g., 3 for 9x9)

    def __hash__(self):
        return hash(self.grid)

    def __eq__(self, other):
        return self.grid == other.grid

    def __str__(self):
        return "".join([str(c) if c != 0 else "." for c in self.grid])

class SudokuProblem(SearchProblem):
    """
    Sudoku implementation of the SearchProblem interface.
    """
    def __init__(self, puzzle_string: str, n: int = 9):
        # Parse string "4.....8.5.3..." to tuple (4, 0, 0, ..., 8, ...)
        grid = []
        for char in puzzle_string:
            if char.isdigit():
                grid.append(int(char))
            elif char == '.':
                grid.append(0)
        self.initial_state = SudokuState(tuple(grid), n)
        self.n = n

    def get_initial_state(self):
        return self.initial_state

    def is_goal(self, state: SudokuState) -> bool:
        return 0 not in state.grid

    def get_successors(self, state: SudokuState) -> List[Tuple[SudokuState, str, float]]:
        """
        Generates successors by finding the first empty cell and trying valid numbers.
        Optimized to pick the variable with fewest legal moves (MRV) to reduce branching factor.
        """
        grid = state.grid
        n = state.n
        
        # 1. Find empty cell with Minimum Remaining Values (MRV)
        best_idx = -1
        best_candidates = None
        min_len = n + 1

        for i in range(n * n):
            if grid[i] == 0:
                candidates = self._get_legal_values(grid, i, n)
                if len(candidates) < min_len:
                    min_len = len(candidates)
                    best_idx = i
                    best_candidates = candidates
                if min_len == 0: break # Unsolvable path
                if min_len == 1: break # Forced move

        if best_idx == -1: return [] # No empty cells or dead end

        successors = []
        for val in best_candidates:
            new_grid = list(grid)
            new_grid[best_idx] = val
            action = f"Place {val} at {best_idx}"
            successors.append((SudokuState(tuple(new_grid), n), action, 1.0))
        
        return successors

    def get_heuristic(self, state: SudokuState) -> float:
        # Heuristic: Number of empty cells (Admissible).
        # Since each step fills 1 cell and cost is 1, h(n) <= true_cost.
        return state.grid.count(0)

    def _get_legal_values(self, grid, idx, n):
        row, col = idx // n, idx % n
        box_size = int(math.sqrt(n))
        box_r, box_c = row // box_size, col // box_size
        
        used = set()
        # Check Row
        for c in range(n): used.add(grid[row * n + c])
        # Check Col
        for r in range(n): used.add(grid[r * n + col])
        # Check Box
        start_r, start_c = box_r * box_size, box_c * box_size
        for r in range(start_r, start_r + box_size):
            for c in range(start_c, start_c + box_size):
                used.add(grid[r * n + c])
        
        return [v for v in range(1, n + 1) if v not in used]

# ==========================================
# PART 2: A* IMPLEMENTATION (Task 2.1)
# ==========================================

class Node:
    def __init__(self, state, parent=None, action=None, g=0.0, h=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

def a_star_search(problem: SearchProblem):
    """
    Generalized A* Search.
    Handles corner cases:
    - Duplicate elimination (Closed Set)
    - Empty frontier (Unsolvable)
    - Lazy expansion (checking duplicates when popping)
    """
    start_time = time.time()
    initial_state = problem.get_initial_state()
    initial_h = problem.get_heuristic(initial_state)
    
    frontier = [] # Priority Queue
    heapq.heappush(frontier, Node(initial_state, None, None, 0, initial_h))
    
    explored = set()
    nodes_expanded = 0
    max_memory = 0

    while frontier:
        max_memory = max(max_memory, len(frontier) + len(explored))
        
        # Pop lowest f-score
        node = heapq.heappop(frontier)
        
        # Corner Case: Duplicate Elimination (No Reopening)
        if node.state in explored:
            continue
        explored.add(node.state)

        # Check Goal
        if problem.is_goal(node.state):
            elapsed = time.time() - start_time
            return {
                "solution": node.state,
                "time": elapsed,
                "nodes_expanded": nodes_expanded,
                "max_memory": max_memory,
                "status": "Solved"
            }

        nodes_expanded += 1

        # Expand
        for next_state, action, cost in problem.get_successors(node.state):
            if next_state not in explored:
                g = node.g + cost
                h = problem.get_heuristic(next_state)
                new_node = Node(next_state, node, action, g, h)
                heapq.heappush(frontier, new_node)

    elapsed = time.time() - start_time
    return {"status": "Unsolvable", "time": elapsed, "nodes_expanded": nodes_expanded}

# ==========================================
# PART 3: SAT SOLVER REDUCTION (Task 2.2)
# ==========================================

class SudokuSAT:
    """
    Solves Sudoku by reducing it to SAT (CNF) and using PySAT.
    """
    def __init__(self, puzzle_string: str, n: int = 9):
        if not PYSAT_AVAILABLE:
            raise ImportError("PySAT not installed. Run 'pip install python-sat'")
        
        self.n = n
        self.grid = [int(c) if c.isdigit() else 0 for c in puzzle_string if c.isdigit() or c == '.']
        self.cnf = CNF()
        self.solver = Glucose3() # A widely used SAT solver

    def _var(self, r, c, v):
        # Map (row, col, value) to a unique integer 1..N^3
        # r, c in 0..8, v in 1..9
        return r * self.n * self.n + c * self.n + v

    def _decode(self, lit):
        # Map integer back to (row, col, value)
        val = lit - 1
        v = (val % self.n) + 1
        val //= self.n
        c = val % self.n
        r = val // self.n
        return r, c, v

    def solve(self):
        start_time = time.time()
        
        # 1. Generate Clauses
        # A. At least one number per cell
        for r in range(self.n):
            for c in range(self.n):
                self.cnf.append([self._var(r, c, v) for v in range(1, self.n + 1)])
                
        # B. At most one number per cell
        for r in range(self.n):
            for c in range(self.n):
                for v1 in range(1, self.n + 1):
                    for v2 in range(v1 + 1, self.n + 1):
                        self.cnf.append([-self._var(r, c, v1), -self._var(r, c, v2)])

        # C. Row, Column, and Box Uniqueness (Standard Sudoku Rules)
        for v in range(1, self.n + 1):
            for k in range(self.n):
                # Row uniqueness
                self.cnf.append([self._var(k, c, v) for c in range(self.n)])
                for c1 in range(self.n):
                    for c2 in range(c1 + 1, self.n):
                        self.cnf.append([-self._var(k, c1, v), -self._var(k, c2, v)])
                
                # Column uniqueness
                self.cnf.append([self._var(r, k, v) for r in range(self.n)])
                for r1 in range(self.n):
                    for r2 in range(r1 + 1, self.n):
                        self.cnf.append([-self._var(r1, k, v), -self._var(r2, k, v)])

        # Box uniqueness
        box_size = int(math.sqrt(self.n))
        for v in range(1, self.n + 1):
            for br in range(box_size):
                for bc in range(box_size):
                    cells = []
                    for r in range(br*box_size, (br+1)*box_size):
                        for c in range(bc*box_size, (bc+1)*box_size):
                            cells.append(self._var(r, c, v))
                    self.cnf.append(cells)
                    for i in range(len(cells)):
                        for j in range(i + 1, len(cells)):
                            self.cnf.append([-cells[i], -cells[j]])

        # D. Pre-filled cells (Unit Clauses)
        for i, val in enumerate(self.grid):
            if val != 0:
                r, c = i // self.n, i % self.n
                self.cnf.append([self._var(r, c, val)])

        # 2. Call Solver
        self.solver.append_formula(self.cnf.clauses)
        is_sat = self.solver.solve()
        elapsed = time.time() - start_time

        result_grid = [0] * (self.n * self.n)
        
        if is_sat:
            model = self.solver.get_model()
            for lit in model:
                if lit > 0:
                    r, c, v = self._decode(lit)
                    result_grid[r * self.n + c] = v
            return {
                "status": "Solved",
                "time": elapsed,
                "solution": "".join(map(str, result_grid)),
                "variables": self.solver.nof_vars(),
                "clauses": self.solver.nof_clauses()
            }
        else:
            return {"status": "Unsolvable", "time": elapsed}

# ==========================================
# EXPERIMENTS
# ==========================================

if __name__ == "__main__":
    # Example "Hard" Puzzle from Norvig's list
    # https://norvig.com/top95.txt
    hard_puzzle = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"
    
    print(f"Problem Instance: {hard_puzzle}")
    print("-" * 40)

    # --- Run A* ---
    print("Running Task 2.1: A* Search...")
    problem = SudokuProblem(hard_puzzle)
    res_astar = a_star_search(problem)
    
    print(f"Status: {res_astar['status']}")
    if res_astar['status'] == 'Solved':
        print(f"Solution: {res_astar['solution']}")
    print(f"Time: {res_astar['time']:.4f}s")
    print(f"Nodes Expanded: {res_astar['nodes_expanded']}")
    print(f"Max Memory (Nodes): {res_astar['max_memory']}")
    print("-" * 40)

    # --- Run SAT ---
    print("Running Task 2.2: Reduction to SAT...")
    if PYSAT_AVAILABLE:
        sat_solver = SudokuSAT(hard_puzzle)
        res_sat = sat_solver.solve()
        
        print(f"Status: {res_sat['status']}")
        if res_sat['status'] == 'Solved':
            print(f"Solution: {res_sat['solution']}")
        print(f"Time: {res_sat['time']:.4f}s")
        print(f"CNF Variables: {res_sat['variables']}")
        print(f"CNF Clauses: {res_sat['clauses']}")
    else:
        print("Skipping SAT execution (PySAT not installed).")