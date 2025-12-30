# AI Homework: Sudoku Solver

### Solving the Sudoku problem using A* and SAT Solver
**Note**: this repository was created for the sole purpose of sharing my code with university professors, all my work was previously done in Google Colab for simplicity and velocity. This is the reason for having just one single commit to the main branch. 

## Short description of the project
The project is divided into 3 main parts:
+ **Generalized code**: The SearchProblem class is an abstract interface. NQueensProblem or GridPathfinding can be implemented by simply inheriting from it and implementing get_successors, is_goal, etc., without changing a single line of the a_star_search function.
+ **A_star**: I implemented the algorithm including
- Duplicate elimination: I used a closed set (explored) to track visited states. This prevents cycles and redundant work, which is critical for graph search.
- Lazy expansion: the check "if node.state in explored: continue" happens after popping from the priority queue. This handles cases where a state is added to the frontier multiple times with different costs; only the best one (first popped) is processed.
- Heuristic: I used the "number of empty cells" as an admissible heuristic. While simple, it guarantees optimality if we treat every step as cost=1.
+ **SAT Reduction**: The code converts Sudoku rules into CNF clauses where
- Cells must have values (1-9).
- Cells cannot have two values.
- Rows, columns, and boxes must have unique values. 
It uses _pysat_ to solve the generated boolean formula. 

## Dependencies
Prerequisites: You will need Python installed. 
For the SAT solver portion, you need the _python-sat_ library. 
To install it run the following command in your terminal:
> pip install python-sat
More simply, to obtain all the dependencies and recreate the same environment I used run:
> pip install -r requirements.txt

## How to run? 
Once the above library is sucessfully installed run the following command in the terminal to run the code:
> python sudoku_solver.py
The script includes a "Main" section that runs a classic Norvig benchmark instance using both algorithms and prints the metrics.
To test the solver with different instances of the problem, just change the main portion of the code at "hard_puzzle=" and run again. 
See the report for further clarifications. 
