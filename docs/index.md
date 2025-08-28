# AutoGPT 文档

欢迎使用 AutoGPT 文档。

AutoGPT 项目包含四个主要组件：

* [Agent](#agent) - AutoGPT 核心代理
* [Benchmark](#benchmark) - 性能测试工具
* [Forge](#forge) - 代理开发框架
* [Frontend](#frontend) - 用户界面

## 🤖 Agent

**[📖 关于 AutoGPT](AutoGPT/index.md)** | **[🔧 安装配置](AutoGPT/setup/index.md)** | **[💻 使用指南](AutoGPT/usage.md)**

基于大语言模型的半自主AI代理，能够执行各种任务。

## 🎯 Benchmark

**[📦 PyPI包](https://pypi.org/project/agbenchmark/)**

标准化的代理性能测试工具，支持 agent protocol 标准，提供客观的性能评估。

## 🏗️ Forge

**[📖 介绍](forge/get-started.md)** | **[🚀 快速开始](https://github.com/Significant-Gravitas/AutoGPT/blob/master/QUICKSTART.md)**

即用型的代理应用模板，提供完整的样板代码，让你专注于代理的核心功能开发。

## 💻 Frontend

**[📘 说明文档](https://github.com/Significant-Gravitas/AutoGPT/blob/master/frontend/README.md)**

开源的用户界面，兼容任何支持 Agent Protocol 的代理。

## 🔧 CLI

项目命令行工具，统一管理所有组件：

```shell
./run agent start autogpt        # 启动 AutoGPT 代理
./run agent create <name>        # 创建新的代理项目
./run benchmark start <agent>    # 运行基准测试
./run setup                      # 安装依赖
```

## 💬 支持

加入 Discord 社区获取帮助：[discord.gg/autogpt](https://discord.gg/autogpt)