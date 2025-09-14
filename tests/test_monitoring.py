import sys
import pathlib
import time

# ensure repository root on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from modules.monitoring import RealTimeMetricsCollector, BottleneckDetector


def test_bottleneck_detection():
    detector = BottleneckDetector(window_size=5)
    collector = RealTimeMetricsCollector(detector)

    for _ in range(5):
        collector.start("fast")
        time.sleep(0.01)
        event_fast = collector.end("fast")
        collector.start("slow")
        time.sleep(0.02)
        event_slow = collector.end("slow")

    # ensure metrics recorded
    assert event_fast.latency > 0
    assert event_fast.throughput > 0
    assert event_slow.energy >= 0

    bottleneck = detector.bottleneck()
    assert bottleneck is not None
    assert bottleneck[0] == "slow"

    # print dashboard for manual inspection
    collector.print_dashboard()
