# 模块说明：modules/brain

`modules/brain` 目录中的实现用于演示和原型验证，当前不会在核心执行流程中自动加载。这里的组件可以作为研究和实验的灵感来源，但在生产环境中仍需手动集成，并进行充分的测试与安全校验。

- ✅ 目的：提供感知、情绪、伦理等子系统的轻量级样例，便于快速试验新的大脑功能。
- ⚠️ 状态：实验性代码，默认不会被 `autogpt.core` 内的代理启用。
- 🛠️ 如何使用：结合 `autogpt.core.brain.encoding` 提供的观测/记忆编码工具，自行桥接需要的模块。
- 📚 训练：使用 `backend/autogpt/autogpt/core/brain/train_transformer_brain.py` 可以对内部 Transformer 大脑进行训练。

如需将此目录下的模块接入正式流程，请确保：
1. 合并后的接口遵循 `BaseAgent` 的观测和记忆编码约定；
2. 在集成前补充单元/集成测试；
3. 评估安全策略、性能开销与日志记录需求。
- Neuromorphic tuning: use \\modules.brain.neuromorphic.tuning.random_search\\ or run \\python -m modules.brain.neuromorphic.tuning --help\\ for random-search experiments. Use `--parallel thread|process` and `--workers` to enable concurrent trials.
- Evaluation helper: run `python -m modules.brain.neuromorphic.evaluate --help` to score configs against recorded targets. Use `--metrics` to choose from `mse` and `total_spikes`.
- Quick start: copy modules/brain/neuromorphic/examples/minimal/ to a workspace, then run tuning/evaluate CLIs with \\--dataset <path>\\ to reuse the bundled config, signal, and target files.
