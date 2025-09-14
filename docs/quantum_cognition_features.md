# Quantum cognition features

This document demonstrates the lightweight quantum cognition helpers used for
unit testing.  They do **not** aim to be physically accurate simulations but
provide minimal behaviour for experiments.

## Quantum decision making

```python
from modules.brain.quantum.quantum_cognition import QuantumCognition

qc = QuantumCognition()
choice, probs = qc.make_decision({
    "A": [0.5, 0.5],
    "B": [0.5, -0.5],
})
# choice == "A", destructive interference removes option B
```

## Concept entanglement with decoherence

```python
from modules.brain.quantum.quantum_cognition import EntanglementNetwork

network = EntanglementNetwork()
density = network.entangle_concepts("cat", "hat", decoherence=0.3)
```

## Quantum memory

```python
from modules.brain.quantum.quantum_cognition import QuantumMemory, SuperpositionState

memory = QuantumMemory()
memory.store("A", SuperpositionState({"0": 1}))
memory.store("B", SuperpositionState({"1": 1}))
combined = memory.superposition({"A": 1/2**0.5, "B": 1/2**0.5})
```
