## CLI 文档

本文档介绍如何使用项目的 CLI（命令行界面），并展示每个命令的预期输出。请注意，`agents stop` 命令会终止运行在 8000 端口上的任何进程。

### 1. CLI 入口

直接运行 `./scripts/run` 不带任何参数会显示帮助信息，其中列出了可用的命令和选项。此外，你可以在任何命令后添加 `--help` 以查看该命令的专属帮助。

```sh
./scripts/run
```

**输出**：

```
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  agent      Commands to create, start and stop agents
  arena      Commands to enter the arena
  benchmark  Commands to start the benchmark and list tests and categories
  setup      Installs dependencies needed for your system.
```

如果你需要任何命令的帮助，只需在命令末尾添加 `--help`，例如：

```sh
./scripts/run COMMAND --help
```

这会显示与该命令相关的详细帮助信息，包括可用的其他选项和参数。

### 2. Setup 命令

```sh
./scripts/run setup
```

**输出**：

```
Setup initiated
Installation has been completed.
```

此命令用于初始化项目的安装。

### 3. Agent 命令

**a. 列出所有代理**

```sh
./scripts/run agent list
```

**输出**：

```
Available agents: 🤖
        🐙 forge
        🐙 autogpt
```

列出所有可用代理。

**b. 创建新代理**

```sh
./scripts/run agent create my_agent
```

**输出**：

```
🎉 New agent 'my_agent' created and switched to the new directory in autogpts folder.
```

创建名为 'my_agent' 的新代理。

**c. 启动代理**

```sh
./scripts/run agent start my_agent
```

**输出**：

```
... (ASCII Art representing the agent startup)
[Date and Time] [forge.sdk.db] [DEBUG] 🐛  Initializing AgentDB with database_string: sqlite:///agent.db
[Date and Time] [forge.sdk.agent] [INFO] 📝  Agent server starting on http://0.0.0.0:8000
```

启动 `my_agent` 并显示启动时的 ASCII 图和日志。

**d. 停止代理**

```sh
./scripts/run agent stop
```

**输出**：

```
Agent stopped
```

停止正在运行的代理。

### 4. Benchmark 命令

**a. 列出 Benchmark 类别**

```sh
./scripts/run benchmark categories list
```

**输出**：

```
Available categories: 📚
        📖 code
        📖 safety
        📖 memory
        ... (and so on)
```

列出所有可用的 Benchmark 类别。

**b. 列出 Benchmark 测试**

```sh
./scripts/run benchmark tests list
```

**输出**：

```
Available tests: 📚
        📖 interface
                🔬 Search - TestSearch
                🔬 Write File - TestWriteFile
        ... (and so on)
```

列出所有可用的 Benchmark 测试。

**c. 显示 Benchmark 测试详情**

```sh
./scripts/run benchmark tests details TestWriteFile
```

**输出**：

```
TestWriteFile
-------------

        Category:  interface
        Task:  Write the word 'Washington' to a .txt file
        ... (and other details)
```

显示 `TestWriteFile` Benchmark 测试的详细信息。

**d. 启动代理的 Benchmark**

```sh
./scripts/run benchmark start my_agent
```

**输出**：

```
(more details about the testing process shown whilst the test are running)
============= 13 failed, 1 passed in 0.97s ============...
```

显示 `my_agent` 的 Benchmark 测试结果。

### 5. Arena 命令

**a. 进入 Arena**

```sh
./scripts/run arena enter my_agent
```

**输出**：

```
🚀 my_agent has entered the arena! Please edit your PR description at the following URL: <PR_URL>
```

将指定代理提交到 Arena，参与排行榜挑战。

### 6. 音频命令

AutoGPT 现在支持在命令系统中进行文本与语音之间的转换：

```sh
text_to_speech "你好，世界"     # 生成语音并保存到工作区
speech_to_text "audio.mp3"     # 将音频文件转录为文本
```

这些命令可在代理的工作流程或技能中调用，用于生成语音回复或处理语音输入。

**b. 更新 Arena 提交**

```sh
./scripts/run arena update my_agent <commit_hash> --branch main
```

**输出**：

```
🚀 The file for agent 'my_agent' has been updated in the arena directory.
```

更新已参赛代理的提交信息，例如新的提交哈希或分支。
