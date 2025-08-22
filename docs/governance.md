# Governance and Approvals

This repository supports a lightweight governance model for coordinating work
between automated agents and human collaborators.

## Roles

Roles and their permissions are declared in [`governance/charter/roles.yaml`](../governance/charter/roles.yaml).
The default charter defines three roles:

- **human_architect** – reviews and approves pull requests.
- **developer** – proposes code changes and responds to feedback.
- **ci_agent** – runs tests and reports status.

## Workflow

1. A developer opens a pull request with proposed changes.
2. Continuous integration agents run automated checks.
3. The human architect reviews the pending commits and records approvals using
   the utilities in [`governance/approvals.py`](../governance/approvals.py).
4. Once all commits are approved, the pull request can be merged.

## Approval Utility

The `ApprovalService` class wraps git commands to track approvals using
[`git notes`](https://git-scm.com/docs/git-notes). Each approval adds a note in
`Approved-by: <name>` format to the corresponding commit. A commit is considered
approved when at least one such note is present.

Example usage:

```python
from governance.approvals import ApprovalService

svc = ApprovalService()
pendings = svc.pending_commits()
for commit in pendings:
    print(commit, "requires approval")
```

Use `add_approval(commit, approver)` to record a new approval and
`is_approved(commit)` to verify approval status.

## Responsibilities

| Role            | Responsibilities                                      |
|-----------------|--------------------------------------------------------|
| human_architect | Validate major changes, ensure safety and governance. |
| developer       | Implement features and incorporate feedback.          |
| ci_agent        | Execute tests and share results.                      |

All participants should follow the governance charter and ensure that changes
receive the necessary approvals before merging.
