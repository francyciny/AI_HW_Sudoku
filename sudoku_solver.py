import time
import heapq
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Any

# Try to import PySAT.
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
    @abstractmethod
    def get_initial_state(self): pass
    @abstractmethod
    def is_goal(self, state) -> bool: pass
    @abstractmethod
    def get_successors(self, state) -> List[Tuple[Any, Any, float]]: pass
    @abstractmethod
    def get_heuristic(self, state) -> float: pass

class SudokuState:
    def __init__(self, grid: Tuple[int], n: int):
        self.grid = grid
        self.n = n
    def __hash__(self): return hash(self.grid)
    def __eq__(self, other): return self.grid == other.grid
    def __str__(self): return "".join([str(c) if c != 0 else "." for c in self.grid])

class SudokuProblem(SearchProblem):
    def __init__(self, puzzle_string: str, n: int = 9):
        grid = []
        for char in puzzle_string:
            if char.isdigit(): grid.append(int(char))
            elif char == '.': grid.append(0)
        self.initial_state = SudokuState(tuple(grid), n)
        self.n = n

    def get_initial_state(self): return self.initial_state
    def is_goal(self, state: SudokuState) -> bool: return 0 not in state.grid

    def get_successors(self, state: SudokuState) -> List[Tuple[SudokuState, str, float]]:
        grid = state.grid
        n = state.n
        
        # MRV Heuristic for Variable Selection
        best_idx, best_candidates, min_len = -1, None, n + 1
        
        for i in range(n * n):
            if grid[i] == 0:
                candidates = self._get_legal_values(grid, i, n)
                if len(candidates) < min_len:
                    min_len = len(candidates)
                    best_idx = i
                    best_candidates = candidates
                if min_len == 0: return [] # Dead end
                if min_len == 1: break # Forced move

        if best_idx == -1: return [] 

        successors = []
        for val in best_candidates:
            new_grid = list(grid)
            new_grid[best_idx] = val
            successors.append((SudokuState(tuple(new_grid), n), f"Place {val}", 1.0))
        return successors

    def get_heuristic(self, state: SudokuState) -> float:
        return state.grid.count(0)

    def _get_legal_values(self, grid, idx, n):
        row, col = idx // n, idx % n
        box_size = int(math.sqrt(n))
        box_r, box_c = row // box_size, col // box_size
        used = set()
        for k in range(n):
            used.add(grid[row * n + k]) 
            used.add(grid[k * n + col]) 
        start_r, start_c = box_r * box_size, box_c * box_size
        for r in range(start_r, start_r + box_size):
            for c in range(start_c, start_c + box_size):
                used.add(grid[r * n + c])
        return [v for v in range(1, n + 1) if v not in used]

# ==========================================
# PART 2: A* IMPLEMENTATION
# ==========================================

class Node:
    def __init__(self, state, parent=None, action=None, g=0.0, h=0.0):
        self.state = state
        self.parent = parent
        self.g = g
        self.f = g + h
    def __lt__(self, other): return self.f < other.f

def a_star_search(problem: SearchProblem, max_nodes=50000):
    start_time = time.time()
    initial_state = problem.get_initial_state()
    frontier = []
    heapq.heappush(frontier, Node(initial_state, None, None, 0, problem.get_heuristic(initial_state)))
    explored = set()
    
    # Metrics
    nodes_expanded = 0
    nodes_generated = 0
    max_memory = 0
    
    # Branching Factor Metrics
    min_b = float('inf')
    max_b = 0
    total_b = 0
    
    while frontier:
        max_memory = max(max_memory, len(frontier) + len(explored))
        
        node = heapq.heappop(frontier)
        
        if node.state in explored: continue
        explored.add(node.state)

        if problem.is_goal(node.state):
            avg_b = total_b / nodes_expanded if nodes_expanded > 0 else 0
            return {
                "status": "Solved",
                "time": time.time()-start_time,
                "nodes_expanded": nodes_expanded,
                "nodes_generated": nodes_generated,
                "max_memory": max_memory,
                "branching": (min_b if min_b != float('inf') else 0, max_b, avg_b),
                "solution": node.state
            }
        
        if nodes_expanded >= max_nodes:
            avg_b = total_b / nodes_expanded if nodes_expanded > 0 else 0
            return {
                "status": "Timeout",
                "time": time.time()-start_time,
                "nodes_expanded": nodes_expanded,
                "nodes_generated": nodes_generated,
                "max_memory": max_memory,
                "branching": (min_b if min_b != float('inf') else 0, max_b, avg_b),
                "solution": None
            }

        nodes_expanded += 1
        successors = problem.get_successors(node.state)
        
        # Branching Metrics
        b_factor = len(successors)
        nodes_generated += b_factor
        total_b += b_factor
        min_b = min(min_b, b_factor)
        max_b = max(max_b, b_factor)

        for next_state, action, cost in successors:
            if next_state not in explored:
                h = problem.get_heuristic(next_state)
                heapq.heappush(frontier, Node(next_state, node, action, node.g + cost, h))

    avg_b = total_b / nodes_expanded if nodes_expanded > 0 else 0
    return {
        "status": "Unsolvable",
        "time": time.time()-start_time,
        "nodes_expanded": nodes_expanded,
        "nodes_generated": nodes_generated,
        "max_memory": max_memory,
        "branching": (min_b if min_b != float('inf') else 0, max_b, avg_b),
        "solution": None
    }

# ==========================================
# PART 3: SAT SOLVER
# ==========================================

class SudokuSAT:
    def __init__(self, puzzle_string: str, n: int = 9):
        self.n = n
        self.grid = [int(c) if c.isdigit() else 0 for c in puzzle_string if c.isdigit() or c == '.']
        if PYSAT_AVAILABLE:
            self.cnf = CNF()
            self.solver = Glucose3()

    def _var(self, r, c, v): return r*self.n*self.n + c*self.n + v
    
    def solve(self):
        if not PYSAT_AVAILABLE: 
            return {"status": "Skipped", "time": 0, "vars": 0, "clauses": 0, "conflicts": 0, "decisions": 0, "propagations": 0}
        
        start_time = time.time()
        
        # 1. Generate Clauses
        # One number per cell
        for r in range(self.n):
            for c in range(self.n):
                self.cnf.append([self._var(r, c, v) for v in range(1, self.n + 1)])
                for v1 in range(1, self.n + 1):
                    for v2 in range(v1 + 1, self.n + 1):
                        self.cnf.append([-self._var(r, c, v1), -self._var(r, c, v2)])

        # Constraints (Row, Col, Box)
        for v in range(1, self.n + 1):
            for k in range(self.n):
                self.cnf.append([self._var(k, c, v) for c in range(self.n)])
                self.cnf.append([self._var(r, k, v) for r in range(self.n)])
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        self.cnf.append([-self._var(k, i, v), -self._var(k, j, v)])
                        self.cnf.append([-self._var(i, k, v), -self._var(j, k, v)])

        b_sz = int(math.sqrt(self.n))
        for v in range(1, self.n + 1):
            for br in range(b_sz):
                for bc in range(b_sz):
                    cells = [self._var(r, c, v) for r in range(br*b_sz, (br+1)*b_sz) for c in range(bc*b_sz, (bc+1)*b_sz)]
                    self.cnf.append(cells)
                    for i in range(len(cells)):
                        for j in range(i + 1, len(cells)):
                            self.cnf.append([-cells[i], -cells[j]])

        # Pre-filled
        for i, val in enumerate(self.grid):
            if val != 0: self.cnf.append([self._var(i//self.n, i%self.n, val)])

        self.solver.append_formula(self.cnf.clauses)
        success = self.solver.solve()
        
        # Get Internal Solver Stats (The new metric!)
        stats = self.solver.accum_stats() 
        # stats is dict like: {'restarts': 1, 'conflicts': 0, 'decisions': 122, 'propagations': 1560}

        res_grid = [0]*81
        if success:
            model = self.solver.get_model()
            for lit in model:
                if lit > 0:
                    val = (lit - 1) % self.n + 1
                    r = (lit - 1) // self.n // self.n
                    c = ((lit - 1) // self.n) % self.n
                    res_grid[r*self.n + c] = val

        return {
            "status": "Solved" if success else "Unsolvable",
            "time": time.time()-start_time,
            "vars": self.solver.nof_vars(),
            "clauses": self.solver.nof_clauses(),
            "conflicts": stats.get('conflicts', 0),
            "decisions": stats.get('decisions', 0),
            "propagations": stats.get('propagations', 0),
            "solution": "".join(map(str, res_grid)) if success else None
        }

def print_grid_pretty(grid_obj, title="Grid"):
    if hasattr(grid_obj, 'grid'): grid_str = "".join(map(str, grid_obj.grid))
    elif isinstance(grid_obj, str): grid_str = grid_obj
    else: return
    print(f"\n--- {title} ---")
    board = [c if c != '0' else '.' for c in grid_str]
    for i in range(9):
        if i % 3 == 0 and i != 0: print("-" * 21)
        row = ""
        for j in range(9):
            if j % 3 == 0 and j != 0: row += "| "
            row += str(board[i * 9 + j]) + " "
        print(row)
    print("-" * 21)

if __name__ == "__main__":
    hard_puzzle = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"
    print("Running Single Instance Test:")
    prob = SudokuProblem(hard_puzzle)
    res = a_star_search(prob)
    print_grid_pretty(prob.initial_state, "Input")
    if res['solution']:
        print_grid_pretty(res['solution'], "Output")
        print(f"Solved in {res['time']:.4f}s with {res['nodes_expanded']} nodes.")
    else:
        print("Could not solve.")