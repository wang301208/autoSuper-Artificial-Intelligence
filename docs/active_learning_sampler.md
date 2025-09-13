# Active Learning Sampler

The `backend/ml/active_sampler.py` module provides a tiny helper for selecting
which new samples should be labelled or used for training. Two strategies are
implemented:

* **Uncertainty** – chooses samples where the model has low confidence.
* **Diversity** – chooses samples that are far from the feature centroid.

## Basic Usage

```python
from backend.ml.active_sampler import ActiveLearningSampler
import numpy as np

sampler = ActiveLearningSampler(strategy="uncertainty")
probs = np.array([[0.6, 0.4], [0.9, 0.1]])
indices = sampler.select(probs=probs, k=1)
# -> array([0]) selects the least certain sample
```

The sampler can be combined with the streaming ingestor in
`backend/runner/streaming.py` to prioritise which events from the queue are
processed:

```python
from backend.runner import StreamingDataIngestor
ingestor = StreamingDataIngestor(sampler=sampler)
ingestor.stream(lambda e: train_on_sample(e))
```
