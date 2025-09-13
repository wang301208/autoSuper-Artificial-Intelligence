# Meta-skill Activation Process

This document describes how meta-skill changes are reviewed and activated.

## Activation Procedure

1. A contributor proposes a change to a meta-skill and opens a meta-ticket using the tooling in `governance/meta_ticket.py`. The ticket records the title, description and pending status; no approval tags are added because activation is automatic.
2. Automated checks run against the proposed change.
3. Once the change is merged, the new meta-skill version is activated automatically in the skill library.

Example meta-ticket JSON:

```json
{
  "title": "Improve search algorithm",
  "description": "Switch to new heuristic",
  "status": "pending"
}
```

## Checklist for Meta-skill Changes

- [ ] Meta-ticket created.
- [ ] Automated tests and reviews have passed.
- [ ] Documentation updated if necessary.

Once all items are complete, the new meta-skill version is active.
