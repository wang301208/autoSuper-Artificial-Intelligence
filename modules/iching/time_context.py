"""Time context utilities for I Ching calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from lunar_python import Solar


@dataclass(frozen=True)
class TimeContext:
    """Represents lunar calendar context information."""

    lunar_year: int
    lunar_month: int
    lunar_day: int
    solar_term: str
    chinese_hour: str


def get_lunar_date(dt: datetime) -> tuple[int, int, int]:
    """Return lunar year, month and day for the given datetime."""

    solar = Solar.fromDate(dt)
    lunar = solar.getLunar()
    return lunar.getYear(), lunar.getMonth(), lunar.getDay()


def get_solar_term(dt: datetime) -> str:
    """Return solar term (\u8282\u6c14) for the given datetime."""

    solar = Solar.fromDate(dt)
    return solar.getLunar().getJieQi()


def get_chinese_hour(dt: datetime) -> str:
    """Return traditional Chinese hour (\u65f6\u8fb0) for the given datetime."""

    solar = Solar.fromDate(dt)
    return solar.getLunar().getTimeZhi()


def get_time_context(dt: datetime) -> TimeContext:
    """Return a complete :class:`TimeContext` for the given datetime."""

    solar = Solar.fromDate(dt)
    lunar = solar.getLunar()
    return TimeContext(
        lunar_year=lunar.getYear(),
        lunar_month=lunar.getMonth(),
        lunar_day=lunar.getDay(),
        solar_term=lunar.getJieQi(),
        chinese_hour=lunar.getTimeZhi(),
    )
