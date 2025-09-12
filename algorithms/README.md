# Algorithm Library

This directory contains algorithm implementations organized by category
and difficulty.

## Language & Style
- Implement algorithms in **Python 3**.
- Follow the **PEP 8** style guide.

## Interface & Naming
- Use the `Algorithm` base class in `base.py`.
- Classes use `PascalCase`.
- Functions and methods use `snake_case`.
- Module and package names use `snake_case`.

## Directory Structure
Algorithms are grouped by their function (e.g., `sorting`, `searching`,
`data_structures`, `storage`, `causal`) and then by difficulty (`basic`, `advanced`).

```
algorithms/
├── sorting/
│   ├── basic/
│   └── advanced/
├── searching/
│   ├── basic/
│   └── advanced/
├── data_structures/
│   ├── basic/
│   └── advanced/
├── storage/
│   ├── basic/
│   └── advanced/
└── causal/
```

Use `template.py` as a starting point for new algorithm implementations.

## Data Structures

Basic examples:

- `algorithms/data_structures/basic/stack.py`
- `algorithms/data_structures/basic/queue.py`
- `algorithms/data_structures/basic/linked_list.py`

Advanced example:

- `algorithms/data_structures/advanced/binary_tree.py`

## Storage

Basic examples:

- `algorithms/storage/basic/lru_cache.py`
- `algorithms/storage/basic/lfu_cache.py`

Advanced example:

- `algorithms/storage/advanced/btree_index.py`

## Causal

Basic example:

- `algorithms/causal/causal_graph.py` - build causal graphs, perform interventions,
  and infer downstream values. See `algorithms/examples.py` for a usage
  demonstration.
