# Governance Charter

AutoGPT agents can be configured with a **governance charter** that defines
which roles exist, what each role may do, and the core directives that guide
all behavior. Charters are stored as JSON or YAML files in
`autogpts/autogpt/data/charter/` and loaded at runtime.

## Format

```yaml
name: Example Charter
core_directives:
  - Serve the user
  - Stay safe
roles:
  - name: assistant
    description: Basic assistant role
    permissions:
      - name: read
        description: Read data sources
      - name: write
        description: Write outputs
    allowed_tasks:
      - answer questions
      - draft texts
```

### Fields

- **name**: A human readable name for the charter.
- **core_directives**: High level guidelines that apply to all roles.
- **roles**: List of roles available in the system.
  - **name**: Unique identifier for the role.
  - **description**: Optional details about the role.
  - **permissions**: Specific permissions granted to the role. Permission
    names must be unique within a role.
  - **allowed_tasks**: Free form descriptions of tasks the role may perform.

## Human Architect Approval

Some tasks may be flagged as *core architecture changes*. These tasks carry
significant risk and must be approved before execution. A charter can define a
`human_architect` role with the `approve_core_change` permission. When a task is
marked with a `core_change` flag (or has type `core_change`), the governance
layer requires approval from this role before routing the task to lower-level
agents.

To grant this authority, include a role similar to:

```yaml
roles:
  - name: human_architect
    description: Human supervisor responsible for core architecture decisions
    permissions:
      - name: approve_core_change
        description: Approve modifications to core system architecture
```

If such approval is not present, the governance agent rejects the task.

## Validation

The loader validates charters using [Pydantic](https://docs.pydantic.dev).
Malformed files or duplicate role/permission names raise a
`CharterValidationError` with details about the problem.

Use `load_charter(name)` to read and validate a charter by filename.
