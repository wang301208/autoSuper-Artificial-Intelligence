# Meta-Learning Examples

This directory contains small CSV datasets that demonstrate how to use the
unified meta-learning trainer. Each dataset provides two numeric feature
columns (`f1`, `f2`) and a `target` column representing the class label.

## Running the trainer

```
python backend/runner/train.py --meta --algorithm maml --config config/meta_learning.yaml
python backend/runner/train.py --meta --algorithm reptile --config config/meta_learning.yaml --shots 2 --ways 2
python backend/runner/train.py --meta --algorithm protonet --config config/meta_learning.yaml --shots 2 --ways 2
```

The configuration file references the datasets in this folder and also
allows specifying the number of shots (examples per class) and ways (classes
per task).
