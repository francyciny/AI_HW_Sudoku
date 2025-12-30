import matplotlib.pyplot as plt
import numpy as np
# IMPORT FROM YOUR SOLVER MODULE
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
        
        # Hardest (Norvig / Inkala)
        ("Expert 1", "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"),
        ("Expert 2", "85...24..72......9..4.........1.7..23.5...9...4...........8..7..17..........36.4."),
        ("Expert 3", "005300000800000020070010500400005300010070006003200080060500009004000030000009700"),
        ("Unsolvable 1", "553020600900305001001806400008102900700000008006708200002609500800203009005010300"), # Conflict at 0,0 and 0,1
        ("Unsolvable 2", "11..............................................................................."),
    ]

    results = {
        "names": [], "astar_time": [], "sat_time": [],
        "astar_expanded": [], "astar_generated": [], "astar_memory": [],
        "b_min": [], "b_max": [], "b_avg": [],
        "sat_vars": [], "sat_clauses": []
    }

    print(f"{'Puzzle':<15} | {'A* Time':<8} | {'SAT Time':<8} | {'Expanded':<8} | {'Status'}")
    print("-" * 75)

    for name, p_str in puzzles:
        results["names"].append(name)
        
        # Run A*
        prob = SudokuProblem(p_str)
        res_a = a_star_search(prob, max_nodes=50000)
        
        results["astar_time"].append(res_a["time"])
        results["astar_expanded"].append(res_a["nodes_expanded"])
        results["astar_generated"].append(res_a["nodes_generated"])
        results["astar_memory"].append(res_a["max_memory"])
        results["b_min"].append(res_a["branching"][0])
        results["b_max"].append(res_a["branching"][1])
        results["b_avg"].append(res_a["branching"][2])

        # Run SAT
        res_s = {"time": 0, "status": "Skipped", "vars": 0, "clauses": 0}
        if PYSAT_AVAILABLE:
            solver = SudokuSAT(p_str)
            res_s = solver.solve()
        
        results["sat_time"].append(res_s["time"])
        results["sat_vars"].append(res_s.get("vars", 0))
        results["sat_clauses"].append(res_s.get("clauses", 0))

        # Output Summary
        status_str = f"A*:{res_a['status']}/SAT:{res_s['status']}"
        print(f"{name:<15} | {res_a['time']:.4f}s  | {res_s['time']:.4f}s  | {res_a['nodes_expanded']:<8} | {status_str}")
        
        # Grid Output for one example (e.g. Expert 1)
        if name == "Expert 1":
            print_grid(prob.initial_state, f"Initial ({name})")
            if res_a['solution']: print_grid(res_a['solution'], "A* Solution")

    return results

def plot_metrics(res):
    names = res["names"]
    x = np.arange(len(names))
    width = 0.35

    # Figure 1: Time & Expanded Nodes
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Time
    ax1.bar(x - width/2, res["astar_time"], width, label='A*', color='skyblue')
    ax1.bar(x + width/2, res["sat_time"], width, label='SAT', color='orange')
    ax1.set_title('Running Time (A* vs SAT)')
    ax1.set_ylabel('Seconds')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_yscale('log') # Log scale helps view both fast and slow runs

    # Nodes
    ax2.bar(x, res["astar_expanded"], width, label='Expanded', color='lightgreen')
    ax2.bar(x, res["astar_generated"], width, label='Generated', color='darkgreen', alpha=0.5, bottom=res["astar_expanded"])
    ax2.set_title('A* Search Graph Size')
    ax2.set_ylabel('Count')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.show()

    # Figure 2: Branching Factor & Memory
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))

    # Branching Factor
    ax3.plot(x, res["b_avg"], marker='o', label='Avg Branching', color='purple')
    ax3.fill_between(x, res["b_min"], res["b_max"], color='purple', alpha=0.1, label='Min-Max Range')
    ax3.set_title('A* Branching Factor (Min/Avg/Max)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.legend()

    # Memory
    ax4.bar(x, res["astar_memory"], color='salmon')
    ax4.set_title('Max Nodes in Memory (Frontier + Explored)')
    ax4.set_ylabel('Count')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # Figure 3: SAT Metrics
    if any(res["sat_vars"]):
        fig3, ax5 = plt.subplots(figsize=(10, 5))
        ax5.scatter(res["sat_vars"], res["sat_clauses"], c='red')
        for i, txt in enumerate(names):
            ax5.annotate(txt, (res["sat_vars"][i], res["sat_clauses"][i]), fontsize=8)
        ax5.set_title('SAT Problem Complexity (Variables vs Clauses)')
        ax5.set_xlabel('Variables')
        ax5.set_ylabel('Clauses')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    results = run_benchmarks()
    plot_metrics(results)