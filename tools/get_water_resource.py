#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9

from values import REGIONS
import pandas as pd

# 加载实测径流量
def get_measured_runoff(region):
    use_cols = {
        '年份': 'Year',
        '唐乃亥': 'SR',  # 唐乃亥控制源区
        '头道拐': 'UR',  # 头道拐控制上游
        '花园口': 'MR',  # 花园口控制中游
        '利津': 'DR'     # 利津控制下游
    }
    measured_runoff = pd.read_csv(r'../data/hydrology/1956-2016_runoff.csv')
    measured_runoff = measured_runoff.loc[:, use_cols.keys()]
    measured_runoff.rename(use_cols, axis=1, inplace=True)
    measured_runoff.set_index('Year', inplace=True)
    return measured_runoff[region]


# 获取天然径流量的差值
def get_runoff_difference(region):
    index = REGIONS.index(region)
    if index == 0:
        runoff_in = 0
    else:
        runoff_in = get_measured_runoff(REGIONS[index-1])
    runoff_out = get_measured_runoff(region)
    return runoff_out - runoff_in


# 加载消耗量
def get_consumptions(region):
    """计算从此区域及其上游区域，合计的【水资源】消耗量，单位是立方千米"""
    city_yr = pd.read_csv(r'../data/perfectures/yr/perfectures_in_YR_with_threshold_0.05.csv')
    water_use = city_yr[['Year', 'Region', 'Total water use']]
    consumption = water_use.groupby(['Region', 'Year']).sum().loc[region]['Total water use']
    flag = 0
    index = REGIONS.index(region)
    while flag < index:
        consumption += water_use.groupby(['Region', 'Year']).sum().loc[REGIONS[flag]]['Total water use']
        flag += 1
    return consumption * 10  # 单位由 10^8 变成 10^9


# 计算每个区域的地表/地下水修正系数
def get_surface_groundwater_coefficient(region):
    watersheds = pd.read_csv(r"../data/watershed_merged.csv")

    def get_type_data(d, p):
        return d.groupby("项目").get_group(p)

    SR = ['龙羊峡以上', '龙羊峡至兰州', '河源-兰州']
    UR = ['兰州至头道拐', '兰州-头道拐']
    MR = ['头道拐至龙门', '龙门至三门峡', '三门峡至花园口', '头道拐-龙门', '龙门-三门峡', '三门峡-花园口']
    DR = ['花园口以下', '花园口-利津', '利津-河口']
    WATERSHED_TO_SUBREGION = {
        "SR": SR,
        "UR": UR,
        "MR": MR,
        "DR": DR
    }

    def judge_region(x):
        for k, v in WATERSHED_TO_SUBREGION.items():
            if x in v:
                return k

            
    watersheds['region'] = watersheds['分区'].apply(judge_region)
    watersheds = watersheds[watersheds['region'].notna()]
    withdraw = get_type_data(watersheds, "取水量")

    def calculate_ratio(data, sector):
        sur = sector + "_surface"
        gro = sector + "_groundwater"
        data[sector+'_sum'] = data[sur] + data[gro]
        data[sector+'_ratio'] = data[gro] / data[sector+'_sum']
        return data

    data = calculate_ratio(withdraw, '合计').groupby(['region', '年份']).mean()['合计_ratio']
    return data.loc[region]