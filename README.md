# AutoGPT - 开源AI代理平台

[![Discord](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoGPT** 是一个开源的AI代理开发平台，让每个人都能构建和使用AI代理。

## 🏗️ 项目结构

```
autogpt-v0.5.1/
├── backend/     # 后端服务和AI代理核心
├── frontend/    # 用户界面和Web应用
├── modules/     # 功能模块和组件
├── config/      # 配置文件
├── docs/        # 项目文档
└── scripts/     # 构建和部署脚本
```

## 🚀 快速开始

### 环境要求
- Python 3.10+
- Flutter 3.x（包含 Dart 3.x）
- Node.js 16+（仅在运行基准测试前端等 Node.js 工具时需要）
- Docker (可选)

### 安装步骤
1. 克隆项目并进入目录
2. 复制 `.env.example` 为 `.env` 并配置API密钥
3. 运行 `python scripts/cli.py setup` 或 `./scripts/run setup` 安装依赖
4. 使用 `python scripts/cli.py agent start <agent-name>` 或 `./scripts/run agent start <agent-name>` 启动代理

### 模块级依赖

部分子模块（例如 `algorithms/`、`modules/`）具有独立的依赖，请在相应目录下
使用以下命令单独安装，以避免全局依赖冲突：

```bash
pip install -r algorithms/requirements.txt
pip install -r modules/requirements.txt
```

**📖 [完整文档](https://docs.agpt.co)** | **⚙️ [配置说明](docs/configuration.md)** | **🚀 [贡献指南](CONTRIBUTING.md)**

## 🔌 硬件支持

支持可插拔的 CPU、GPU 和 TPU 后端，详见 [硬件配置文档](docs/hardware_backends.md)。

## 🧱 核心组件

### 🏗️ Forge - 代理开发框架
即用型的代理应用模板，提供完整的样板代码和SDK组件。
- 🚀 [快速入门](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/forge/tutorials/001_getting_started.md)
- 📘 [详细文档](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpts/forge)

### 🎯 Benchmark - 性能测试
支持agent protocol的标准化基准测试工具，提供客观的性能评估。
- 📦 [PyPI包](https://pypi.org/project/agbenchmark/)
- 🏆 [排行榜](https://leaderboard.agpt.co)

### 💻 用户界面
基于 Flutter 构建的用户友好 Web 界面，通过 agent protocol 与代理连接，支持多种代理兼容。
- 📘 [前端文档](https://github.com/Significant-Gravitas/AutoGPT/tree/master/frontend)

### ⌨️ 命令行工具
统一的CLI工具，支持代理管理、基准测试等功能：

```bash
python scripts/cli.py agent start <agent-name>    # 启动代理
python scripts/cli.py benchmark start             # 运行基准测试
python scripts/cli.py setup                       # 安装依赖

# 或使用封装脚本
./scripts/run agent start <agent-name>
./scripts/run benchmark start
./scripts/run setup
```

### 🎤 音频能力

内置 `text_to_speech` 与 `speech_to_text` 命令，可在代理任务或技能中调用，实现文本与语音之间的互相转换。

## 💬 支持与反馈

- **Discord社区**: [加入讨论](https://discord.gg/autogpt)
- **问题报告**: [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)

## 🔄 技术标准

采用 [Agent Protocol](https://agentprotocol.ai/) 标准，确保与各种应用的兼容性。