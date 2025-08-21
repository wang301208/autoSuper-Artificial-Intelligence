# Self-upgrading Agents

AutoGPT agents can now modify their own source code and verify those
changes automatically. Two built-in abilities make this possible:

1. **write_file** – create or overwrite files in the agent's workspace.
2. **run_tests** – execute the project's test suite using `pytest`.

A typical self-upgrade cycle works as follows:

1. Use `write_file` to add or change code in the workspace.
2. Invoke `run_tests` to run the relevant tests.
3. Store the test results in memory and critique the change.
4. Keep the modification if the tests pass, otherwise revert the file.

These abilities allow an agent to iteratively improve itself while
ensuring that new behaviour is validated by automated tests.

