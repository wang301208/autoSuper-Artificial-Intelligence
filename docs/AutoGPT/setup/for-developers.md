# Developer setup & guide

This page walks through setting up a development environment for AutoGPT and
highlights useful tooling for debugging and contributing.

## Local development

### Prerequisites
- **Python 3.10+**
- **Poetry** for dependency management
- **Git**
- *(Optional)* **Docker** for running services such as Redis

### Getting started
1. Fork and clone the repository.
2. Copy the example env file and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
3. Install dependencies:
   ```bash
   poetry install
   ```
4. Set up pre-commit hooks to run linting automatically:
   ```bash
   poetry run pre-commit install
   ```
5. Run the CLI in development mode:
   ```bash
   poetry run python -m autogpt --help
   ```

### Debugging locally
- Set `LOG_LEVEL=DEBUG` in `.env` to enable verbose logs.
- Use `poetry run pytest` to run unit tests.
- When using VSCode or another IDE, you can add breakpoints and launch the
  module `autogpt` directly with the debugger.

## Development with DevContainer

We provide a ready-to-use [DevContainer](https://code.visualstudio.com/docs/devcontainers/containers) for VS Code.

1. Install [VS Code](https://code.visualstudio.com/),
   the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
   and [Docker](https://www.docker.com/get-started).
2. Open the repository in VS Code and click **Reopen in Container** when prompted.
3. The container installs dependencies automatically; once ready you can run and
debug AutoGPT from the integrated terminal.
4. To update dependencies inside the container run `poetry install`.

### Debugging in the container
- Use VS Code's **Run and Debug** panel and select the preconfigured launch
  tasks.
- Logs from the application appear in the integrated terminal; adjust the
  `LOG_LEVEL` in `.env` for more detail.

## FAQ

### Poetry command not found
Ensure Poetry is [installed](https://python-poetry.org/docs/#installation) and
added to your shell PATH. Restart your shell after installation.

### Tests fail with missing API keys
AutoGPT requires an OpenAI key for many tests. Set `OPENAI_API_KEY` in your `.env`
file or export it before running `pytest`.

### Docker cannot start on macOS/Windows
Confirm that Docker Desktop is running. Rebooting the application often resolves
resource allocation errors.

## Recommended resources
- [VS Code documentation](https://code.visualstudio.com/docs)
- [Dev Container specification](https://containers.dev/)
- [Python debugging with VS Code](https://code.visualstudio.com/docs/python/debugging)
- [Poetry documentation](https://python-poetry.org/docs/)
- [OpenAI API docs](https://platform.openai.com/docs)

With a working environment you can explore the rest of the codebase and start
contributing. Make sure to read the [contributing guide](../../contributing.md)
for coding standards and pull request expectations.
