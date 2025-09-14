class PrimaryMotor:
    """Primary motor cortex responsible for executing motor commands."""

    def execute(self, motor_command: str) -> str:
        return f"executed {motor_command}"


class PreMotorArea:
    """Pre-motor area that plans movements based on intention."""

    def plan(self, intention: str) -> str:
        return f"plan for {intention}"


class SupplementaryMotor:
    """Supplementary motor area that coordinates complex movements."""

    def organize(self, plan: str) -> str:
        return f"organized {plan}"


class MotorCortex:
    """Motor cortex integrating multiple motor-related areas."""

    def __init__(
        self,
        basal_ganglia=None,
        cerebellum=None,
        spiking_backend=None,
        ethics=None,
    ):
        self.primary_motor = PrimaryMotor()
        self.premotor_area = PreMotorArea()
        self.supplementary_motor = SupplementaryMotor()
        self.basal_ganglia = basal_ganglia
        self.cerebellum = cerebellum
        self.spiking_backend = spiking_backend
        # Optional ethical reasoning engine used to vet actions before
        # execution.  The engine is expected to expose an ``evaluate_action``
        # method returning a compliance report.
        self.ethics = ethics

    def plan_movement(self, intention: str) -> str:
        """Plan a movement based on an intention."""
        plan = self.premotor_area.plan(intention)
        plan = self.supplementary_motor.organize(plan)
        if self.basal_ganglia:
            plan = self.basal_ganglia.modulate(plan)
        return plan

    def execute_action(self, motor_command: str):
        """Execute a motor command, optionally using a spiking backend."""
        # Perform ethical evaluation before any physical execution.
        if self.ethics:
            report = self.ethics.evaluate_action(motor_command)
            if not report["compliant"]:
                return report

        if self.spiking_backend and isinstance(motor_command, (list, tuple)):
            spikes = self.spiking_backend.run(motor_command)
            self.spiking_backend.synapses.adapt(
                self.spiking_backend.spike_times, self.spiking_backend.spike_times
            )
            return spikes
        command = motor_command
        if self.cerebellum:
            command = self.cerebellum.fine_tune(command)
        return self.primary_motor.execute(command)

    def train(self, error_signal: str) -> str:
        """Adjust motor output based on feedback using the cerebellum."""
        if self.cerebellum:
            return self.cerebellum.motor_learning(error_signal)
        return f"no cerebellum to learn from {error_signal}"
