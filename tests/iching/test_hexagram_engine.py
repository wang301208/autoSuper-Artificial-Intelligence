from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.iching.hexagram_engine import HexagramEngine
from modules.iching.time_context import get_time_context


def test_hexagram_engine_time_context_affects_interpretation():
    engine = HexagramEngine()
    hexagram = "乾"

    ctx_summer = get_time_context(datetime(2023, 6, 21, 12, 0, 0))
    ctx_winter = get_time_context(datetime(2023, 12, 22, 23, 0, 0))

    result_summer = engine.interpret(hexagram, ctx_summer)
    result_winter = engine.interpret(hexagram, ctx_winter)

    assert ctx_summer.solar_term != ctx_winter.solar_term
    assert ctx_summer.chinese_hour != ctx_winter.chinese_hour
    assert result_summer != result_winter
    assert ctx_summer.solar_term in result_summer
    assert ctx_winter.solar_term in result_winter
