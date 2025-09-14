"""Definitions of the sixty-four hexagrams and an interpretation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from .bagua import PreHeavenBagua, PostHeavenBagua
from .time_context import TimeContext


@dataclass(frozen=True)
class Hexagram:
    """Represents a hexagram with its associated texts."""

    number: int
    name: str
    chinese: str
    judgement: str
    lines: Tuple[str, str, str, str, str, str]
    upper: str
    lower: str


# Basic information for the 64 hexagrams.
# Each tuple: (number, English name, Chinese, upper trigram, lower trigram)
_HEXAGRAM_INFO = [
    (1, "Qian", "乾", "Qian", "Qian"),
    (2, "Kun", "坤", "Kun", "Kun"),
    (3, "Zhun", "屯", "Kan", "Zhen"),
    (4, "Meng", "蒙", "Gen", "Kan"),
    (5, "Xu", "需", "Kan", "Qian"),
    (6, "Song", "讼", "Qian", "Kan"),
    (7, "Shi", "师", "Kan", "Kun"),
    (8, "Bi", "比", "Kun", "Kan"),
    (9, "Xiao Chu", "小畜", "Xun", "Qian"),
    (10, "Lu", "履", "Qian", "Dui"),
    (11, "Tai", "泰", "Kun", "Qian"),
    (12, "Pi", "否", "Qian", "Kun"),
    (13, "Tong Ren", "同人", "Qian", "Li"),
    (14, "Da You", "大有", "Li", "Qian"),
    (15, "Qian", "谦", "Kun", "Gen"),
    (16, "Yu", "豫", "Zhen", "Kun"),
    (17, "Sui", "随", "Dui", "Zhen"),
    (18, "Gu", "蛊", "Gen", "Xun"),
    (19, "Lin", "临", "Kun", "Dui"),
    (20, "Guan", "观", "Xun", "Kun"),
    (21, "Shi He", "噬嗑", "Li", "Zhen"),
    (22, "Bi", "贲", "Gen", "Li"),
    (23, "Bo", "剥", "Gen", "Kun"),
    (24, "Fu", "复", "Kun", "Zhen"),
    (25, "Wu Wang", "无妄", "Qian", "Zhen"),
    (26, "Da Chu", "大畜", "Gen", "Qian"),
    (27, "Yi", "颐", "Gen", "Zhen"),
    (28, "Da Guo", "大过", "Dui", "Xun"),
    (29, "Kan", "坎", "Kan", "Kan"),
    (30, "Li", "离", "Li", "Li"),
    (31, "Xian", "咸", "Dui", "Gen"),
    (32, "Heng", "恒", "Zhen", "Xun"),
    (33, "Dun", "遯", "Qian", "Gen"),
    (34, "Da Zhuang", "大壮", "Zhen", "Qian"),
    (35, "Jin", "晋", "Li", "Kun"),
    (36, "Ming Yi", "明夷", "Kun", "Li"),
    (37, "Jia Ren", "家人", "Xun", "Li"),
    (38, "Kui", "睽", "Li", "Dui"),
    (39, "Jian", "蹇", "Kan", "Gen"),
    (40, "Jie", "解", "Zhen", "Kan"),
    (41, "Sun", "损", "Gen", "Dui"),
    (42, "Yi", "益", "Xun", "Zhen"),
    (43, "Guai", "夬", "Dui", "Qian"),
    (44, "Gou", "姤", "Qian", "Xun"),
    (45, "Cui", "萃", "Dui", "Kun"),
    (46, "Sheng", "升", "Kun", "Xun"),
    (47, "Kun", "困", "Dui", "Kan"),
    (48, "Jing", "井", "Kan", "Xun"),
    (49, "Ge", "革", "Dui", "Li"),
    (50, "Ding", "鼎", "Li", "Xun"),
    (51, "Zhen", "震", "Zhen", "Zhen"),
    (52, "Gen", "艮", "Gen", "Gen"),
    (53, "Jian", "渐", "Xun", "Gen"),
    (54, "Gui Mei", "归妹", "Zhen", "Dui"),
    (55, "Feng", "丰", "Zhen", "Li"),
    (56, "Lü", "旅", "Li", "Gen"),
    (57, "Xun", "巽", "Xun", "Xun"),
    (58, "Dui", "兑", "Dui", "Dui"),
    (59, "Huan", "涣", "Xun", "Kan"),
    (60, "Jie", "节", "Kan", "Dui"),
    (61, "Zhong Fu", "中孚", "Xun", "Dui"),
    (62, "Xiao Guo", "小过", "Zhen", "Gen"),
    (63, "Ji Ji", "既济", "Kan", "Li"),
    (64, "Wei Ji", "未济", "Li", "Kan"),
]


_hexagrams = []

# Detailed text for Hexagram 1 - Qian
_hexagrams.append(
    Hexagram(
        number=1,
        name="Qian",
        chinese="乾",
        judgement="The Creative works sublime success, furthering through perseverance.",
        lines=(
            "Hidden dragon. Do not act.",
            "Dragon appearing in the field.",
            "All day long the superior man is creatively active.",
            "Wavering flight over the depths.",
            "Flying dragon in the heavens.",
            "A dragon overreaches himself.",
        ),
        upper="Qian",
        lower="Qian",
    )
)

# Detailed text for Hexagram 2 - Kun
_hexagrams.append(
    Hexagram(
        number=2,
        name="Kun",
        chinese="坤",
        judgement="The Receptive brings about sublime success, furthering through perseverance.",
        lines=(
            "When there is hoarfrost underfoot, solid ice is not far off.",
            "Straight, square, great. Without purpose, yet nothing remains unfurthered.",
            "Hidden lines. One is able to remain persevering.",
            "A tied-up sack. No blame, no praise.",
            "A yellow lower garment brings supreme good fortune.",
            "Dragons fight in the meadow; their blood is black and yellow.",
        ),
        upper="Kun",
        lower="Kun",
    )
)

# Populate remaining hexagrams with placeholder text
for number, name, chinese, upper, lower in _HEXAGRAM_INFO[2:]:
    _hexagrams.append(
        Hexagram(
            number=number,
            name=name,
            chinese=chinese,
            judgement=f"Judgement text for hexagram {number}",
            lines=tuple(
                f"Line {i + 1} of hexagram {number}" for i in range(6)
            ),
            upper=upper,
            lower=lower,
        )
    )

# Mapping from (upper, lower) trigram names to hexagram data
def _build_name_map() -> Dict[Tuple[str, str], Hexagram]:
    mapping: Dict[Tuple[str, str], Hexagram] = {}
    for hexagram in _hexagrams:
        key = (hexagram.upper.lower(), hexagram.lower.lower())
        mapping[key] = hexagram
    return mapping


HEXAGRAM_MAP = _build_name_map()


class HexagramEngine:
    """Simple engine to fetch hexagram interpretations by trigram pairing."""

    def __init__(self) -> None:
        self._map = HEXAGRAM_MAP
        # Map possible trigram inputs (English/Chinese) to canonical English names
        self._trigram_names: Dict[str, str] = {}
        for bagua_cls in (PreHeavenBagua, PostHeavenBagua):
            for trigram in bagua_cls:
                t = trigram.value
                self._trigram_names[t.name.lower()] = t.name
                self._trigram_names[t.chinese] = t.name

    def _normalize(self, name: str) -> str:
        normalized = name.strip().lower()
        if normalized not in self._trigram_names:
            raise KeyError(f"Unknown trigram '{name}'")
        return self._trigram_names[normalized]

    def interpret(
        self, upper: str, lower: str, time_ctx: Optional[TimeContext] = None
    ) -> Hexagram:
        """Return the hexagram for the given upper and lower trigrams.

        If ``time_ctx`` is provided, the judgement and line texts are annotated
        with contextual information from the lunar calendar, solar term, and
        shichen.
        """

        upper_name = self._normalize(upper)
        lower_name = self._normalize(lower)
        key = (upper_name.lower(), lower_name.lower())
        if key not in self._map:
            raise KeyError(f"Combination ({upper}, {lower}) not found")
        hexagram = self._map[key]
        if not time_ctx:
            return hexagram

        extras = []
        if time_ctx.solar_term:
            extras.append(f"Solar term: {time_ctx.solar_term}")
        extras.append(f"Shichen: {time_ctx.shichen}")
        judgement = f"{hexagram.judgement} ({'; '.join(extras)})"
        lines = tuple(f"{line} ({time_ctx.shichen})" for line in hexagram.lines)
        return Hexagram(
            number=hexagram.number,
            name=hexagram.name,
            chinese=hexagram.chinese,
            judgement=judgement,
            lines=lines,
            upper=hexagram.upper,
            lower=hexagram.lower,
        )

