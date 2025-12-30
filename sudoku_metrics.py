import matplotlib.pyplot as plt
import numpy as np
from sudoku_solver import SudokuProblem, SudokuSAT, a_star_search, PYSAT_AVAILABLE, print_grid

def run_benchmarks():
    puzzles = [
        # Easy
        ("Easy 1", "003020600900305001001806400008102900700000008006708200002609500800203009005010300"),
        ("Easy 2", "200080300060070084030500209000105408000000000402706000301007040720040060004010003"),
        ("Easy 3", "000000907000420180000705026100904000050000040000507009920108000034059000507000000"),
        # Medium
        ("Medium 1", "000200063300005401001003980000000090000538000030000000026300500503700008470001000"),
        ("Medium 2", "020608000580009700000040000370000500600000004008000013000020000009800036000306090"),
        ("Medium 3", "000604700706000009000005080070020093800000005430010070050200000300000208002301000"),
        # Hard
        ("Hard 1", "000600400700003600000091080000000000050180003000306045040200060903000000020000100"),
        ("Hard 2", "200300000804062003013800200000020390507000621032006000020009140601250809000001002"),
        ("Hard 3", "000000000000003085001020000000507000004000100090000000500000073002010000000040009"),
        # Expert
        ("Expert 1", "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"),
        ("Expert 2", "85...24..72......9..4.........1.7..23.5...9...4...........8..7..17..........36.4."),
        ("Expert 3", "005300000800000020070010500400005300010070006003200080060500009004000030000009700"),
        # Corner cases
        ("Unsolvable 1", "553020600900305001001806400008102900700000008006708200002609500800203009005010300"),
        ("Unsolvable 2", "11..............................................................................."),
    ]

    results = {
        "names": [], "clues": [],
        "astar_time": [], "sat_time": [],
        "astar_expanded": [], "astar_memory": [],
        "b_min": [], "b_max": [], "b_avg": [],
        "sat_vars": [], "sat_clauses": [],
        "sat_conflicts": [], "sat_decisions": []
    }

    print(f"{'Puzzle':<15} | {'Clues':<5} | {'A* Time':<8} | {'SAT Time':<8} | {'Expanded':<8} | {'Status'}")
    print("-" * 85)

    for name, p_str in puzzles:
        results["names"].append(name)
        
        # Calculate clues (problem density: number of filled cells at initial state)
        clues = sum(1 for c in p_str if c.isdigit() and c != '0')
        results["clues"].append(clues)
        
        # Run A*
        prob = SudokuProblem(p_str)
        res_a = a_star_search(prob, max_nodes=50000)
        
        results["astar_time"].append(res_a["time"])
        results["astar_expanded"].append(res_a["nodes_expanded"])
        results["astar_memory"].append(res_a["max_memory"])
        results["b_min"].append(res_a["branching"][0])
        results["b_max"].append(res_a["branching"][1])
        results["b_avg"].append(res_a["branching"][2])

        # Run SAT
        res_s = {"time": 0, "status": "Skipped", "vars": 0, "clauses": 0, "conflicts": 0, "decisions": 0}
        if PYSAT_AVAILABLE:
            solver = SudokuSAT(p_str)
            res_s = solver.solve()
        
        results["sat_time"].append(res_s["time"])
        results["sat_vars"].append(res_s.get("vars", 0))
        results["sat_clauses"].append(res_s.get("clauses", 0))
        results["sat_conflicts"].append(res_s.get("conflicts", 0))
        results["sat_decisions"].append(res_s.get("decisions", 0))

        status_str = f"A*:{res_a['status']}/SAT:{res_s['status']}"
        print(f"{name:<15} | {clues:<5} | {res_a['time']:.4f}s  | {res_s['time']:.4f}s  | {res_a['nodes_expanded']:<8} | {status_str}")
        
        if name == "Expert 1":
            print_grid(prob.initial_state, f"Initial ({name})")
            if res_a['solution']: print_grid(res_a['solution'], "A* Solution")

    return results

def plot_metrics(res):
    names = res["names"]
    x = np.arange(len(names))
    width = 0.35

    # Time & Expanded Nodes
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Time
    ax1.bar(x - width/2, res["astar_time"], width, label='A*', color='skyblue')
    ax1.bar(x + width/2, res["sat_time"], width, label='SAT', color='orange')
    ax1.set_title('Running Time (A* vs SAT)')
    ax1.set_ylabel('Seconds')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_yscale('log')

    # A* Search Space
    ax2.bar(x, res["astar_expanded"], width, label='Nodes Expanded', color='lightgreen')
    ax2.set_title('A* Search Effort (Nodes Expanded)')
    ax2.set_ylabel('Count')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.show()

    # Branching & Memory
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))
    ax3.plot(x, res["b_avg"], marker='o', label='Avg Branching', color='purple')
    ax3.fill_between(x, res["b_min"], res["b_max"], color='purple', alpha=0.1, label='Min-Max Range')
    ax3.set_title('A* Branching Factor')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.legend()

    ax4.bar(x, res["astar_memory"], color='salmon')
    ax4.set_title('Max Memory Usage (Nodes)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # SAT Detailed Metrics
    if any(res["sat_vars"]):
        fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Work Done (Conflicts & Decisions)
        ax5.bar(x - width/2, res["sat_decisions"], width, label='Decisions', color='gold')
        ax5.bar(x + width/2, res["sat_conflicts"], width, label='Conflicts', color='red')
        ax5.set_title('SAT Solver "Effort" (Decisions vs Conflicts)')
        ax5.set_xticks(x)
        ax5.set_xticklabels(names, rotation=45, ha='right')
        ax5.legend()
        ax5.set_yscale('log')

        # Difficulty vs Time
        clues = np.array(res["clues"])
        times = np.array(res["astar_time"])
        ax6.scatter(clues, times, c='blue', s=100, alpha=0.7)
        for i, txt in enumerate(names):
            ax6.annotate(txt, (clues[i], times[i]), fontsize=9)
        ax6.set_title('Puzzle Difficulty: Initial Clues vs A* Time')
        ax6.set_xlabel('Number of Initial Clues (Density)')
        ax6.set_ylabel('A* Time (s)')
        ax6.set_yscale('log')
        ax6.grid(True, which="both", ls="--")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    results = run_benchmarks()
    plot_metrics(results)