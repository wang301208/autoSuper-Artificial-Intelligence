# Strategy Evolution

The `EvolutionAgent` layer uses a policy-gradient method to mutate and select abilities
based on feedback from executed tasks. Training data and the learned policy are
stored under `skills/MetaSkill_StrategyEvolution`:

- `policy.json` – current policy weights for each state and ability
- `training_log.json` – history of state/action/reward tuples used during training

## Dataset formats

Training data can also be loaded from external datasets. The `PolicyTrainer`
accepts CSV, JSON, or JSONL files where each record defines a state, action, and
reward. Records may be nested, and a custom `experience_transform` function can
be supplied to map arbitrary schemas into the required `(state, action, reward)`
tuples.

## Configuration

The agent can be tuned with environment variables:

- `EVOLUTION_LEARNING_RATE` – learning rate for policy updates (default: `0.1`)
- `EVOLUTION_GENERATIONS` – number of generations before a mutation step (default: `10`)
- `EVOLUTION_FITNESS_FUNCTION` – fitness function used to score results (default: `reward`)

After each generation the strategy template is updated automatically from the
collected training data.
