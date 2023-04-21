#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Shuang (Twist) Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Website: https://cv.songshgeo.com/

import numpy as np
import pandas as pd

from regimes_yrb.tools.statistic import pettitt_changes, ratio_contribution


def calculate_yearly_indices(region_consumption: pd.DataFrame, year: int) -> dict:
    """
    计算指定年份的水资源分配熵指数。

    参数:
    region_consumption (pd.DataFrame): 按年份和区域分组的用水量数据
    year (int): 需要计算指数的年份

    返回:
    dict: 该年份的水资源分配熵指数数据，包含以下键值：
          - "Regions": 区域间熵指数
          - "Divisions": 部门间熵指数
          - "Sectors": 总体熵指数
          - "Ratio": 区域间与部门间熵指数之比
    """
    data = region_consumption.loc[year].replace(0.0, 0.0000000001)
    region_p = data.sum(axis=1) / data.sum().sum()
    region_entropy = np.sum(-region_p * np.log(region_p))

    # section_entropy = (
    #     data.apply(lambda row: row / row.sum(), axis=0)
    #     .apply(lambda row: row.apply(lambda p: -p * np.log(p)).sum(), axis=0)
    #     .mean()
    # )
    # section_max_p = data.sum() / data.sum().sum()
    # section_max_entropy = section_max_p.apply(lambda p: -p * np.log(p)).sum()
    # # section_index = section_entropy / section_max_entropy

    index_ratio = region_entropy / -np.log(1 / 4)

    return {
        "Regions": region_entropy,
        # "Divisions": section_entropy,
        # "Sectors": section_max_entropy,
        "Ratio": index_ratio,
    }


def calc_regional_entropy(city_yr: pd.DataFrame) -> pd.DataFrame:
    """
    计算每年各区域的总用水，并计算区域间、部门间以及总体的水资源分配熵指数。

    参数:
    city_yr (pd.DataFrame): 城市年度用水数据，包含以下列：
                             - "Year": 年份
                             - "Region": 区域
                             - "IRR": 农业用水
                             - "IND": 工业用水
                             - "RUR": 农村生活用水
                             - "URB": 城市生活用水

    返回:
    pd.DataFrame: 每年的水资源分配熵指数数据，包含以下列：
                  - "Regions": 区域间熵指数
                  - "Divisions": 部门间熵指数
                  - "Sectors": 总体熵指数
                  - "Ratio": 区域间与部门间熵指数之比
    """
    wu_cols = ["IRR", "IND", "DOM"]
    city_yr["DOM"] = city_yr["RUR"] + city_yr["URB"]  # 人居耗水
    region_consumption = city_yr.groupby(["Year", "Region"])[wu_cols].sum()  # 每个区域的总耗水量

    results = {
        yr: calculate_yearly_indices(region_consumption, yr)
        for yr in city_yr["Year"].unique()
    }
    return pd.DataFrame(results).T["Ratio"]


# def entropy_and_contribution(data):
#     """熵指数和各地区的贡献"""
#     index_result = calculate_water_consumption_indices(data)
#     entropy_contributions = ratio_contribution(
#         numerator=index_result[["Sectors", "Regions"]],
#         denominator=index_result["Divisions"],
#         breakpoints=pettitt_changes(index_result["Ratio"]),
#     )
#     return index_result["Ratio"], entropy_contributions
