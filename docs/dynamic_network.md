# 自适应结构的动态神经网络

`DynamicNetwork` 模块展示了一种简单的增量学习策略：
当训练损失的改善低于阈值时，模型自动增加新的隐藏层；
当损失显著恶化时，则移除最近添加的层。

## 设计原理

- **增长触发**：连续若干个 epoch 内损失改善不足 `growth_trigger`，
  并且当前层数未达到 `max_layers`，则添加一个新的隐藏层。
- **剪枝触发**：连续若干个 epoch 内损失变差超过 `shrink_trigger`，
  则移除最近的隐藏层以防止过拟合。
- **Patience**：`patience` 控制触发条件需持续的 epoch 数。

## 使用方式

```python
from algorithms.neuro_symbolic import DynamicConfig, DynamicNetwork
import numpy as np

cfg = DynamicConfig(max_layers=3, growth_trigger=0.05, shrink_trigger=-0.1, patience=2)
net = DynamicNetwork(input_size=4, output_size=1, config=cfg)

# 假设 data 和 targets 为 numpy 数组
output = net.execute(data, targets, epochs=10)
```

配置参数也可以通过 `config/dynamic_network.yaml` 加载和修改，
以便限制模型容量，避免无限膨胀。
