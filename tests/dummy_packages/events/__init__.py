class EventBus:
    def publish(self, *args, **kwargs):
        pass

    def subscribe(self, *args, **kwargs):
        pass

def create_event_bus(*args, **kwargs):
    return EventBus()

def publish(bus, *args, **kwargs):
    bus.publish(*args, **kwargs)

def subscribe(bus, *args, **kwargs):
    bus.subscribe(*args, **kwargs)
