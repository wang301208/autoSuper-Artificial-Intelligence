from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.iching.time_context import (
    get_chinese_hour,
    get_lunar_date,
    get_solar_term,
    get_time_context,
)


def test_time_context_summer_solstice_noon():
    dt = datetime(2023, 6, 21, 12, 0, 0)
    year, month, day = get_lunar_date(dt)
    assert (year, month, day) == (2023, 5, 4)
    assert get_solar_term(dt) == "夏至"
    assert get_chinese_hour(dt) == "午"


def test_time_context_winter_solstice_midnight():
    dt = datetime(2023, 12, 22, 23, 0, 0)
    ctx = get_time_context(dt)
    assert ctx.solar_term == "冬至"
    assert ctx.chinese_hour == "子"
