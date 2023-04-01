#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

from typing import Iterable, LiteralString, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .tools.statistic import (
    plot_pittitt_change_points,
    plot_ratio_contribution,
    ratio_contribution,
)

Columns: TypeAlias = Iterable[LiteralString]


def direct_ratio(
    data: pd.DataFrame, direct: Columns, indirect: Columns
) -> pd.DataFrame:
    """计算直接用水比例"""
    # 间接用水量
    direct_wu = data.loc[:, direct]
    # 总用水量
    total_water_use = data.loc[:, [*direct, *indirect]].sum(
        axis=1, numeric_only=True
    )
    # 整个流域的间接惠益用水占比
    priority = direct_wu.sum(axis=1) / total_water_use
    # 计算贡献
    contributions = ratio_contribution(
        numerator=direct_wu, denominator=total_water_use
    )
    return priority, contributions


# def plot_priority(priority, contributions):
#     # 绘图
#     fig = plt.figure(figsize=(8, 3), constrained_layout=True)
#     gs = GridSpec(1, 2, figure=fig, width_ratios=[3, 4])
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])

#     # 绘制图1
#     plot_pittitt_change_points(
#         priority, change_points=[1977, 1993], ax=ax1, colors=period_colors
#     )

#     # 绘制图2
#     plot_ratio_contribution(
#         contributions,
#         ax=ax2,
#         colors=region_colors,
#         denominator_color=total_water_use_color,
#         denominator_label="Total",
#     )

#     # 修饰图1
#     ax1.set_ylabel("Indirect Proportion of Water Use, configuration")
#     ax1.set_xlabel("Year")
#     # ax1.set_yticks(np.arange(0.08, 0.21, 0.04))
#     # ax1.set_ylim(0.08, 0.21)
#     # ax1.text(1968, 0.2, 'a.', ha='center', va='center', weight='bold', size='large')

#     # 修饰图2
#     ax2.set_xlabel("Different periods")
#     ax2.set_ylabel("Changes of configuration")
#     ax2.tick_params(axis="x", tickdir="in", bottom=False, labelrotation=0)
#     ax2.axhline(0.0, lw=2, color="gray")
#     ax2.set_ylim(-0.08, 0.08)
#     ax2.set_yticks(np.arange(-0.08, 0.081, 0.04))
#     ax2.axvline(1.5, ls=":", color="gray", lw=1.5)
#     ax2.axvline(0.5, ls=":", color="gray", lw=1.5)
#     ax2.text(
#         2.4, 0.065, "b.", ha="center", va="center", weight="bold", size="large"
#     )

#     # 调整坐标轴显示
#     for ax in [ax1, ax2]:
#         ax.spines["top"].set_visible(False)
#         ax.spines["bottom"].set_visible(False)
#         ax.spines["left"].set_visible(True)
#         ax.spines["right"].set_visible(False)
#     ax1.spines["bottom"].set_visible(True)

#     # 出图
#     plt.show()
