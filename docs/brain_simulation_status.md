# Brain Simulation Status

## Current Coverage
- `modules/brain/whole_brain.py:1` describes the integration as a simplified wiring demo rather than a faithful whole brain simulation, signalling missing physiological breadth.
- `modules/brain/whole_brain.py:150` exposes only cortical sensory, limbic, cognitive, and motor structures; deeper subcortical loops, cerebellar routing, and oscillation control objects exist in the package but are not orchestrated here.
- `modules/brain/motor_cortex.py:43` keeps cerebellar and spiking backends optional, and the default `WholeBrainSimulation` never primes these hooks, so motor refinement and neuromorphic execution are dormant.
- `tests/brain/test_whole_brain_neuromorphic.py:1` exercises a basic single-cycle happy path but skips multi-region coordination, plasticity, and oscillatory validation.

## Gaps Against "完整的大脑模拟+神经形态"
- Missing closed-loop integration across cerebellum, basal ganglia, oscillatory dynamics, and neuromorphic motor control.
- No persistent neuromorphic backend shared between perception and motor planning; perception spikes stop at modality snapshots.
- Lacks explicit wiring to oscillation generators for synchrony, energy budgeting, or cognitive state gating.
- Absent durability and evaluation flows to tune neuromorphic parameters against representative activity traces.

## Implementation Starting Point
1. Embed cerebellar fine-tuning and shared spiking backend into `WholeBrainSimulation` so motor plans run through neuromorphic pathways before dispatch.
2. Drive oscillation synthesis (alpha/beta/gamma loops) off neuromorphic perception metrics and expose the resulting synchrony to downstream agents.
3. Extend metrics and state exports to capture oscillation, cerebellar learning, and neuromorphic load, and cover them with regression tests.
4. Stage follow-up work: multi-region memory consolidation, persistent plasticity checkpoints, and dataset-driven neuromorphic tuning.
