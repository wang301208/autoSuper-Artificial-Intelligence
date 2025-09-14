"""Example demonstrating brain regions communicating via the message bus."""
from modules.brain.message_bus import publish_neural_event, subscribe_to_brain_region


def motor_cortex(event: dict) -> None:
    """Handle motor commands."""
    print(f"Motor cortex received: {event['command']}")


def visual_cortex() -> None:
    """Publish a visual signal that triggers a motor response."""
    publish_neural_event("visual_signal", {"command": "wave"})


def main() -> None:
    subscribe_to_brain_region("motor", "visual_signal", motor_cortex)
    visual_cortex()


if __name__ == "__main__":
    main()
