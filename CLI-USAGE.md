## CLI 文档

本文档介绍如何使用项目的 CLI（命令行界面），并展示每个命令的预期输出。请注意，`agents stop` 命令会终止运行在 8000 端口上的任何进程。

### 1. CLI 入口

直接运行 `./run` 不带任何参数会显示帮助信息，其中列出了可用的命令和选项。此外，你可以在任何命令后添加 `--help` 以查看该命令的专属帮助。

```sh
./run
```

**输出**：

```
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  agent      Commands to create, start and stop agents
  benchmark  Commands to start the benchmark and list tests and categories
  setup      Installs dependencies needed for your system.
```

如果你需要任何命令的帮助，只需在命令末尾添加 `--help`，例如：

```sh
./run COMMAND --help
```

这会显示与该命令相关的详细帮助信息，包括可用的其他选项和参数。

### 2. Setup 命令

```sh
./run setup
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
./run agent list
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
./run agent create my_agent
```

**输出**：

```
🎉 New agent 'my_agent' created and switched to the new directory in autogpts folder.
```

创建名为 'my_agent' 的新代理。

**c. 启动代理**

```sh
./run agent start my_agent
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
./run agent stop
```

**输出**：

```
Agent stopped
```

停止正在运行的代理。

### 4. Benchmark 命令

**a. 列出 Benchmark 类别**

```sh
./run benchmark categories list
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
./run benchmark tests list
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
./run benchmark tests details TestWriteFile
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
./run benchmark start my_agent
```

**输出**：

```
(more details about the testing process shown whilst the test are running)
============= 13 failed, 1 passed in 0.97s ============...
```

显示 `my_agent` 的 Benchmark 测试结果。
