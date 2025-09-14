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

    def __init__(self, basal_ganglia=None, cerebellum=None):
        self.primary_motor = PrimaryMotor()
        self.premotor_area = PreMotorArea()
        self.supplementary_motor = SupplementaryMotor()
        self.basal_ganglia = basal_ganglia
        self.cerebellum = cerebellum

    def plan_movement(self, intention: str) -> str:
        """Plan a movement based on an intention."""
        plan = self.premotor_area.plan(intention)
        plan = self.supplementary_motor.organize(plan)
        if self.basal_ganglia:
            plan = self.basal_ganglia.modulate(plan)
        return plan

    def execute_action(self, motor_command: str) -> str:
        """Execute a motor command, optionally refined by the cerebellum."""
        command = motor_command
        if self.cerebellum:
            command = self.cerebellum.fine_tune(command)
        return self.primary_motor.execute(command)
