# AutoGPT 文档

欢迎使用 AutoGPT 文档。

AutoGPT 项目包含四个主要组件：

* [Agent](#agent) - AutoGPT 核心代理
* [Benchmark](#benchmark) - 性能测试工具
* [Forge](#forge) - 代理开发框架
* [Frontend](#frontend) - 用户界面
* [Algorithms](#algorithms) - 常见算法与数据结构实现
* [API Reference](#api-reference) - AutoGPT Python 接口文档

## 🤖 Agent

**[📖 关于 AutoGPT](AutoGPT/index.md)** | **[🔧 安装配置](AutoGPT/setup/index.md)** | **[💻 使用指南](AutoGPT/usage.md)**

基于大语言模型的半自主AI代理，能够执行各种任务。

## 🎯 Benchmark

**[📦 PyPI包](https://pypi.org/project/agbenchmark/)**

标准化的代理性能测试工具，支持 agent protocol 标准，提供客观的性能评估。

## 🏗️ Forge

**[📖 介绍](forge/get-started.md)** | **[🚀 快速开始](../QUICKSTART.md)**

即用型的代理应用模板，提供完整的样板代码，让你专注于代理的核心功能开发。

## 💻 Frontend

**[📘 说明文档](../frontend/README.md)**

开源的用户界面，兼容任何支持 Agent Protocol 的代理。

## 📚 Algorithms

**[📘 使用说明](../algorithms/README.md)**

提供排序、搜索、数据结构以及存储与缓存等常见算法实现，例如 `LRUCache`、`LFUCache`、`BTreeIndex` 等。新增的 `DynamicNetwork` 模块支持在训练过程中动态增删隐藏层。

## 📖 API Reference

**[📘 API 参考](api.md)**

AutoGPT Python 包的 API 文档。

## 依赖管理

为避免全局依赖冲突，某些子模块（如 `algorithms/`、`modules/`）提供独立的依赖文件。可在模块目录下执行以下命令单独安装：

```bash
pip install -r <module>/requirements.txt
```

项目同时提供 `ModernDependencyManager` 来按需解析并安装缺失依赖，内部通过 `importlib` 与 `pip` 等官方接口实现，可自动处理版本范围并减少对弃用 API 的依赖。这样可以仅安装所需依赖并保持环境整洁。

## 🔧 CLI

项目命令行工具，统一管理所有组件：

```shell
./scripts/run agent start autogpt        # 启动 AutoGPT 代理
./scripts/run agent create <name>        # 创建新的代理项目
./scripts/run benchmark start <agent>    # 运行基准测试
./scripts/run setup                      # 安装依赖
```

## 🌅 未来展望

想了解 AutoGPT 的长期研究方向，请参阅 [未来展望](agi_vision_cn.md)。

## 💬 支持

加入 Discord 社区获取帮助：[discord.gg/autogpt](https://discord.gg/autogpt)
