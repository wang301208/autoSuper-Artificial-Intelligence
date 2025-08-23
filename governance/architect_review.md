# System Architect Review Process

This document describes how meta-skill changes are reviewed and approved.

## Approval Procedure

1. A contributor proposes a change to a meta-skill and opens a meta-ticket using the tooling in `governance/meta_ticket.py`.
2. The ticket is automatically tagged **awaiting-system-architect-approval** and announced through the project's normal communication channels.
3. The System Architect reviews the ticket and related code changes.
4. Upon approval, the architect updates the ticket status and activates the meta-skill version in the skill library.

## Checklist for Meta-skill Changes

- [ ] Meta-ticket created and tagged as awaiting System Architect approval.
- [ ] Implementation reviewed and approved by the System Architect.
- [ ] Meta-skill activated via `SkillLibrary.activate_meta_skill`.
- [ ] Documentation updated if necessary.

Only after all items are complete may the new meta-skill version be considered active.
