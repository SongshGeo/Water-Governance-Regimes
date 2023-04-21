#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from typing import Iterable, List, LiteralString, Optional, TypeAlias

import pandas as pd

from .tools.statistic import pettitt_changes, ratio_contribution

Columns: TypeAlias = Iterable[LiteralString]


def direct_ratio(
    data: pd.DataFrame,
    direct: Columns,
    indirect: Columns,
    breakpoints: Optional[List[int]] = None,
) -> pd.DataFrame:
    """计算直接用水比例"""
    # 间接用水量
    direct_wu = data.loc[:, direct]
    # 总用水量
    total_water_use = data.loc[:, [*direct, *indirect]].sum(axis=1, numeric_only=True)
    # 整个流域的间接惠益用水占比
    priority = direct_wu.sum(axis=1) / total_water_use
    if breakpoints is None:
        breakpoints = pettitt_changes(priority)
    else:
        breakpoints = list(breakpoints)
    # 计算贡献
    contributions = ratio_contribution(
        numerator=direct_wu,
        denominator=total_water_use,
        breakpoints=breakpoints,
    )
    return priority, contributions
