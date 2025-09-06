# Breadth-First Search

**Purpose:** Traverse a graph level by level.

**Input:**
- `graph`: `Graph` instance
- `start`: starting node

**Output:** `List[Any]` of nodes visited in BFS order.

## Example
```python
from algorithms.graph.basic.bfs import BreadthFirstSearch
from algorithms.utils import Graph

g = Graph()
g.add_edge('A', 'B')
g.add_edge('A', 'C')
g.add_edge('B', 'D')

BreadthFirstSearch().execute(g, 'A')
```
Output:
```
['A', 'B', 'C', 'D']
```
