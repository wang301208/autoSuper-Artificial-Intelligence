# AutoGPT：构建与使用 AI 代理

[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoGPT** 让每个人都能使用和构建 AI。我们的使命是提供工具，让你专注于真正重要的事情：

- 🏗️ **构建** - 为惊人的创意打下基础。
- 🧪 **测试** - 将你的代理调到完美。
- 🤝 **委派** - 让 AI 为你工作，使你的想法成真。

成为革命的一部分！**AutoGPT** 将始终站在 AI 创新的前沿。

**📖 [文档](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [贡献](CONTRIBUTING.md)**
&ensp;|&ensp;
**🛠️ [构建你的代理 - 快速开始](QUICKSTART.md)**

## 🥇 当前最佳代理：evo.ninja
[Current Best Agent]: #-current-best-agent-evoninja

AutoGPT Arena 黑客松中，[**evo.ninja**](https://github.com/polywrap/evo.ninja) 在我们的 Arena 排行榜上夺得第一，证明了自己是最佳的开源通用代理。现在就到 https://evo.ninja 试试吧！

📈 想挑战 evo.ninja、AutoGPT 等代理？提交你的基准测试到[排行榜](#-leaderboard)，也许下一个上榜的就是你的代理！

## 🧱 构建模块

### 🏗️ Forge

**打造你的专属代理！** – Forge 是一个即用型的代理应用模板。所有样板代码都已处理好，让你把精力集中在使*你的*代理与众不同的部分。[教程在这里](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec)。来自 [`forge.sdk`](/autogpts/forge/forge/sdk) 的组件也可单独使用，加快开发速度并减少样板代码。

🚀 [**Forge 入门**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/forge/tutorials/001_getting_started.md) – 本指南将引导你创建自己的代理，并使用基准测试和用户界面。

📘 [了解更多](https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpts/forge) 关于 Forge 的信息。

### 🎯 Benchmark

**衡量你的代理性能！** `agbenchmark` 可用于任何支持 agent protocol 的代理，并与项目的 [CLI] 集成，使其在 AutoGPT 和基于 Forge 的代理中更易使用。该基准测试提供严格的测试环境。我们的框架允许自主、客观的性能评估，确保你的代理为现实应用做好准备。

<!-- TODO: insert visual demonstrating the benchmark -->

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [了解更多](https://github.com/Significant-Gravitas/AutoGPT/blob/master/benchmark) 关于 Benchmark

#### 🏆 [排行榜][leaderboard]
[leaderboard]: https://leaderboard.agpt.co

通过界面提交你的基准测试结果并在 AutoGPT Arena 排行榜上占据一席之地！得分最高的通用代理将获得 **[当前最佳代理]** 的称号，并被收录到我们的仓库中，方便人们通过 [CLI] 运行。

![AutoGPT Arena leaderboard 的截图](https://github.com/Significant-Gravitas/AutoGPT/assets/12185583/60813392-9ddb-4cca-bb44-b477dbae225d)

### 💻 UI

**让代理使用更简单！** `frontend` 为你提供了一个用户友好的界面来控制和监控代理。它通过[agent protocol](#-agent-protocol) 与代理连接，确保与生态内外的多种代理兼容。

<!-- TODO: instert screenshot of front end -->

前端与仓库中的所有代理开箱即用。只需使用 [CLI] 运行你选择的代理即可！

📘 [了解更多](https://github.com/Significant-Gravitas/AutoGPT/tree/master/frontend) 关于前端的内容。

### ⌨️ CLI

[CLI]: #-cli

为尽可能方便地使用仓库提供的所有工具，根目录中包含一个 CLI：

```shell
$ ./run
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  agent      Commands to create, start and stop agents
  arena      Commands to enter the arena
  benchmark  Commands to start the benchmark and list tests and categories
  setup      Installs dependencies needed for your system.
```

只需克隆仓库，使用 `./run setup` 安装依赖即可开始！

### 🤖 自动代码生成

AutoGPT 包含 `generate_code` 指令，可根据你的提示生成代码并将结果写入代理的工作区。

```json
{"command": {"name": "generate_code", "args": {"prompt": "编写一个打印 \"Hello, world!\" 的 Python 程序"}}}
```

生成的代码会保存为工作区中的新文件（文件名包含时间戳）。
**限制**：该代码由 LLM 自动生成，未经过审查或执行，可能包含错误或安全隐患，运行前请务必检查。

## 🤔 有问题？遇到困难？有建议？

### 获取帮助 - [Discord 💬](https://discord.gg/autogpt)

![Join us on Discord](https://invidget.switchblade.xyz/autogpt)

要报告错误或请求功能，请创建一个 [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose)。请确保没有其他人已经为同一主题创建 Issue。

## 🛡️ Meta-skill 治理

Meta-skill 的变更需由 System Architect 审核。使用 `governance/meta_ticket.py` 创建 meta-ticket；它会被自动标记为 `awaiting-system-architect-approval` 并通过现有沟通渠道通知。更多细节见 [governance/architect_review.md](governance/architect_review.md)。

## 姐妹项目

### 🔄 Agent Protocol

为保持统一标准并确保与众多现有和未来应用的无缝兼容，AutoGPT 采用 AI Engineer Foundation 的 [agent protocol](https://agentprotocol.ai/) 标准。该标准规范了从你的代理到前端和基准测试的通信路径。

---

<p align="center">
<a href="https://star-history.com/#Significant-Gravitas/AutoGPT">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
  </picture>
</a>
</p>
