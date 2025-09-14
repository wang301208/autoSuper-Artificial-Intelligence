# AutoGPT Agent User Guide

!!! note
    This guide assumes you are in the `autogpts/autogpt` folder, where the AutoGPT Agent
    is located.

## Command Line Interface

Running `./autogpt.sh` (or any of its subcommands) with `--help` lists all the possible
sub-commands and arguments you can use:

```shell
$ ./autogpt.sh --help
Usage: python -m autogpt [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  run    Sets up and runs an agent, based on the task specified by the...
  serve  Starts an Agent Protocol compliant AutoGPT server, which creates...
```

!!! important "For Windows users"
    On Windows, use `.\autogpt.bat` instead of `./autogpt.sh`.
    Everything else (subcommands, arguments) should work the same.

!!! info "Usage with Docker"
    For use with Docker, replace the script in the examples with
    `docker compose run --rm auto-gpt`:

    ```shell
    docker compose run --rm auto-gpt --ai-settings <filename>
    docker compose run --rm auto-gpt serve
    ```

### `run` &ndash; CLI mode

The `run` sub-command starts AutoGPT with the legacy CLI interface.

<details>
<summary>
<code>./autogpt.sh run --help</code>
</summary>

```shell
$ ./autogpt.sh run --help
Usage: python -m autogpt run [OPTIONS]

  Sets up and runs an agent, based on the task specified by the user, or
  resumes an existing agent.

Options:
  -c, --continuous                Enable Continuous Mode
  -y, --skip-reprompt             Skips the re-prompting messages at the
                                  beginning of the script
  -C, --ai-settings FILE          Specifies which ai_settings.yaml file to
                                  use, relative to AutoGPT's config directory.
                                  Will also automatically skip the re-prompt.
  -P, --prompt-settings FILE      Specifies which prompt_settings.yaml file to
                                  use, relative to AutoGPT's config directory.
  -l, --continuous-limit INTEGER  Defines the number of times to run in
                                  continuous mode
  --speak                         Enable Speak Mode
  --debug                         Enable Debug Mode
  --gpt3only                      Enable GPT3.5 Only Mode
  --gpt4only                      Enable GPT4 Only Mode
  -b, --browser-name TEXT         Specifies which web-browser to use when
                                  using selenium to scrape the web.
  --allow-downloads               Dangerous: Allows AutoGPT to download files
                                  natively.
  --skip-news                     Specifies whether to suppress the output of
                                  latest news on startup.
  --install-plugin-deps           Installs external dependencies for 3rd party
                                  plugins.
  --ai-name TEXT                  AI name override
  --ai-role TEXT                  AI role override
  --constraint TEXT               Add or override AI constraints to include in
                                  the prompt; may be used multiple times to
                                  pass multiple constraints
  --resource TEXT                 Add or override AI resources to include in
                                  the prompt; may be used multiple times to
                                  pass multiple resources
  --best-practice TEXT            Add or override AI best practices to include
                                  in the prompt; may be used multiple times to
                                  pass multiple best practices
  --override-directives           If specified, --constraint, --resource and
                                  --best-practice will override the AI's
                                  directives instead of being appended to them
  --help                          Show this message and exit.
```
</details>

This mode allows running a single agent, and saves the agent's state when terminated.
This means you can *resume* agents at a later time. See also [agent state].

!!! note
    For legacy reasons, the CLI will default to the `run` subcommand when none is
    specified: running `./autogpt.sh run [OPTIONS]` does the same as `./autogpt.sh [OPTIONS]`,
    but this may change in the future.

#### 💀 Continuous Mode ⚠️

Run the AI **without** user authorization, 100% automated.
Continuous mode is NOT recommended.
It is potentially dangerous and may cause your AI to run forever or carry out actions you would not usually authorize.
Use at your own risk.

```shell
./autogpt.sh --continuous
```

To exit the program, press ++ctrl+c++

### `serve` &ndash; Agent Protocol mode with UI

With `serve`, the application exposes an Agent Protocol compliant API and serves a
frontend, by default on `http://localhost:8000`. You can configure the port it is served on with the `AP_SERVER_PORT` environment variable.

<details>
<summary>
<code>./autogpt.sh serve --help</code>
</summary>

```shell
$ ./autogpt.sh serve --help
Usage: python -m autogpt serve [OPTIONS]

  Starts an Agent Protocol compliant AutoGPT server, which creates a custom
  agent for every task.

Options:
  -P, --prompt-settings FILE  Specifies which prompt_settings.yaml file to
                              use, relative to AutoGPT's config directory.
  --debug                     Enable Debug Mode
  --gpt3only                  Enable GPT3.5 Only Mode
  --gpt4only                  Enable GPT4 Only Mode
  -b, --browser-name TEXT     Specifies which web-browser to use when using
                              selenium to scrape the web.
  --allow-downloads           Dangerous: Allows AutoGPT to download files
                              natively.
  --install-plugin-deps       Installs external dependencies for 3rd party
                              plugins.
  --help                      Show this message and exit.
```
</details>

For more information about the API of the application, see [agentprotocol.ai](https://agentprotocol.ai).

### 前端使用

AutoGPT 提供一个基于 Flutter 的 Web 前端，默认由 `./autogpt.sh serve` 命令在
`http://localhost:8000` 提供。你也可以在本地单独构建并调试该前端以便开发。

#### 环境准备

1. 安装 Flutter 3.x（附带 Dart 3.x），参见 [Flutter 官方安装文档](https://docs.flutter.dev/get-started/install)。
2. 进入仓库的 `frontend` 目录。
3. 获取依赖：

   ```bash
   flutter pub get
   ```

4. 在 Linux 使用 Chromium 时，需将 `CHROME_EXECUTABLE` 指向 Chromium 可执行文件，例如：

   ```bash
   export CHROME_EXECUTABLE=/usr/bin/chromium
   ```

#### 构建

* Web 版本：`flutter build web`，构建产物位于 `build/web`，可配合 `./autogpt.sh serve` 一同部署。
* 其他平台：根据目标平台使用 `flutter build macos`、`flutter build windows`、`flutter build apk` 等命令。

#### 运行

* **后端模式**：运行 `./autogpt.sh serve` 后，在浏览器访问 `http://localhost:8000` 即可使用前端。
* **开发模式**：在 `frontend` 目录运行下列命令，然后在浏览器访问 `http://localhost:5000`：

  ```bash
  flutter run -d chrome --web-port 5000
  ```

#### 常见问题

* **找不到 Chrome 可执行文件**：在 Linux 上设置 `CHROME_EXECUTABLE` 环境变量指向 Chromium 路径。
* **`flutter pub get` 失败或缓慢**：检查网络连接，可使用国内镜像源。
* **前端无法连接后端**：确保已经执行 `./autogpt.sh serve` 并确认端口未被防火墙或其他进程占用。

#### 协作面板

前端提供“协作面板”以便多人实时观察与干预代理执行过程。面板实时展示：

* 当前代理计划
* 构建中的世界模型
* 性能指标

用户可以在面板底部文本框中输入补充知识或计划修正，并通过 **Inject** 或 **Correct** 按钮发送。所有内容通过 WebSocket 与后端双向同步，支持多客户端协同编辑与提示推送。

### Arguments

!!! attention
    Most arguments are equivalent to configuration options. See [`.env.template`][.env.template]
    for all available configuration options.

!!! note
    Replace anything in angled brackets (<>) to a value you want to specify

Here are some common arguments you can use when running AutoGPT:

* Run AutoGPT with a different AI Settings file

    ```shell
    ./autogpt.sh --ai-settings <filename>
    ```

* Run AutoGPT with a different Prompt Settings file

    ```shell
    ./autogpt.sh --prompt-settings <filename>
    ```

!!! note
    There are shorthands for some of these flags, for example `-P` for `--prompt-settings`.  
    Use `./autogpt.sh --help` for more information.

[.env.template]: https://github.com/Significant-Gravitas/AutoGPT/tree/master/autogpts/autogpt/.env.template

## Agent State
[agent state]: #agent-state

The state of individual agents is stored in the `data/agents` folder. You can use this
in various ways:

* Resume your agent at a later time.
* Create "checkpoints" for your agent so you can always go back to specific points in
    its history.
* Share your agent!

## Workspace
[workspace]: #workspace

Agents can read and write files. This happens in the `workspace` folder, which
is in `data/agents/<agent_id>/`. Files outside of this folder can not be accessed by the
agent *unless* `RESTRICT_TO_WORKSPACE` is set to `False`.

!!! warning
    We do not recommend disabling `RESTRICT_TO_WORKSPACE`, unless AutoGPT is running in
    a sandbox environment where it couldn't do any damage (e.g. Docker or a VM).

## Logs

Activity, Error, and Debug logs are located in `logs`.

!!! tip
    Do you notice weird behavior with your agent? Do you have an interesting use case? Do you have a bug you want to report?
    Follow the step below to enable your logs. You can include these logs when making an issue report or discussing an issue with us.

To print out debug logs:

```shell
./autogpt.sh --debug
```

## Disabling Command Categories

If you want to selectively disable some command groups, you can use the
`DISABLED_COMMAND_CATEGORIES` config in your `.env`. You can find the list of available
categories [here][command categories].

For example, to disable coding related features, set it to the value below:

```ini
DISABLED_COMMAND_CATEGORIES=autogpt.commands.execute_code
```

[command categories]: https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpts/autogpt/autogpt/commands/__init__.py
