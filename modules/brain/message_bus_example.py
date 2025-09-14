"""Example of brain regions communicating over the message bus."""
from __future__ import annotations

from modules.brain.message_bus import (
    publish_neural_event,
    reset_message_bus,
    subscribe_to_brain_region,
)


def run_example() -> None:
    reset_message_bus()
    log: list[str] = []

    def motor_handler(event):
        log.append(f"motor received: {event['payload']}")

    def visual_handler(event):
        # Visual cortex decides to instruct the motor cortex
        publish_neural_event({"target": "motor", "payload": event["payload"]})

    subscribe_to_brain_region("visual", visual_handler)
    subscribe_to_brain_region("motor", motor_handler)

    publish_neural_event({"target": "visual", "payload": "move arm"})
    print("\n".join(log))


if __name__ == "__main__":
    run_example()
