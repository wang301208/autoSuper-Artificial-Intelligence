# Depth-First Search

**Purpose:** Traverse a graph exploring as far as possible along each branch before backtracking.

**Input:**
- `graph`: `Graph` instance
- `start`: starting node

**Output:** `List[Any]` of nodes visited in DFS order.

## Example
```python
from algorithms.graph.basic.dfs import DepthFirstSearch
from algorithms.utils import Graph

g = Graph()
g.add_edge('A', 'B')
g.add_edge('A', 'C')
g.add_edge('B', 'D')

DepthFirstSearch().execute(g, 'A')
```
Output:
```
['A', 'B', 'D', 'C']
```
