#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append(r"..")
from tools.processing import get_region_by_province_name
from tools.values import PROVINCES, INDUSTRIES_eng, INDUSTRIES, REGIONS



def get_yr_gdp_data(how='gdp'):
    # 清洗列名
    def clean_col_names(name):
        elements = name.split(':')  # 分割列名
        province = elements[0].strip()  # 地名
        industry = elements[-1].strip()  # 产业
        return province, industry
        
    # 加载经济数据
    gdp = (pd.read_excel('../data/GDP.xlsx')  # 读取数据
            .drop(0)  # 删除单位行
            .astype(float)  # 转为浮点
            .rename({'指标名称': 'Year'}, axis=1)  # 将年份列改名
            .astype({'Year': int})  # 将年份从浮点改为整形
            .set_index('Year')  # 并设置为索引
            .replace(0., np.nan)  # 空值是没数据，不可能为 0
            .drop(list(np.arange(1949, 1965)), axis=0)  # 删除不考虑的时间范围
            )
    region_dict = {region: pd.DataFrame() for region in REGIONS}  # 用来储存每个区域的GDP数据
    for col in gdp:
        province = clean_col_names(col)[0]  # 清洗列名
        region = get_region_by_province_name(province)  # 属于哪个区域
        if region: region_dict[region][col] = gdp[col]  # 如果是黄河流域的省份，存进对应的字典
        else:  gdp.drop(col, axis=1, inplace=True)  # 不是黄河流域的省份，就丢掉

    if how == 'gdp':
        return gdp
    else:
        return region_dict



# 根据年份，提取数据
def extract_data_by_yr(data, start_yr, end_yr):
    # 当需要提取的年份在起点年份之前，采用时间序列中最早的一年
    if start_yr not in data.index:
        start_yr = data.index[0]
    # 同理，如果终点年份超出索引，采用最晚的一年
    if end_yr not in data.index:
        end_yr = data.index[-1]

    # 当两者相等时，无标准差
    if start_yr == end_yr:
        return data.loc[end_yr], False
    else:
        use_data = data.loc[start_yr: end_yr]
        return np.mean(use_data), np.std(use_data)


def plot_bar_by_category(ax, data, categories, colors, position, start_yr, end_yr, **kargs):
    """ 绘制一个分类柱状图，将不同类别画在同一个位置上
    ax: 绘图区域
    data: 数据，DataFrame，index 是时间（年份），列与 categories 相关
    categories: 分类列表，其中的类应该是 data 中出现的列名
    colors: 每个类使用的颜色
    position: 这个分类柱状图的柱子位置
    start_yr: 截取数据的起始年
    end_yr: 截取数据的末位年
    **kargs: 与绘图相关的参数
    ---------------------------------------------
    return: 返回图例
    """
    bottom = 0  # 初始的柱状图底
    legends = []  # 图例
    for i, category in enumerate(categories):  # 对每个分类
        use_data = data[category]  # 数据是其中的类
        height, std = extract_data_by_yr(use_data, start_yr, end_yr)
        if std:  # 如果有标准差，就绘制误差线
            bar = ax.bar(
                x=position,
                bottom=bottom,
                height=height, 
                yerr=std,
                color=colors[i],
                **kargs
            )
        else:
            bar = ax.bar(
                x=position,
                bottom=bottom,
                height=height, 
                color=colors[i],
                **kargs
            )
        # 每次循环之后：
        bottom += height  # 原先的高作为新的底
        legends.append(bar)  # 图像存为图例
    return legends


# 根据指定的产业提取数据，计算区域总的 GDP
def extract_gdp_by_industry(data, industry):
    """ 根据指定的产业，提取GDP数据
    data: GDP 数据，列名格式为："地区: GDP: 第X产业"
    industry: 需要提取的产业
    --------------------------------
    return: 给定数据集中，所有该产业的逐年 GDP
    """
    df = pd.DataFrame()
    for col in data:
        if industry in col:
            df[col] = data[col]
        else:
            continue
    return df.sum(axis=1)


# GDP 制图
def plot_gdp(ax, colors, start_yr, end_yr, **kargs):
    region_dict = get_yr_gdp_data(how='dic')
    # 对每个区域进行处理
    for i, region in enumerate(REGIONS):
        use_data = region_dict[region]  # 某个区域的数据
        industries_gdp = []  # 用来储存该区域的 一、二、三 产业的总平均 gdp
        # INDUSTRIES_std = []  # 用来储存该区域的 一、二、三 产业的gdp方差

        gdp_region_data = {}
        # 对每个产业进行处理
        for industry in INDUSTRIES:
            # 该区域、该产业的gdp序列，以产业为键存入字典，以供函数调用
            gdp_region_data[industry] = extract_gdp_by_industry(use_data, industry)

        legends = plot_bar_by_category(
            ax=ax,
            data=gdp_region_data,
            categories=INDUSTRIES, 
            position=i,
            start_yr=start_yr,
            end_yr=end_yr,
            colors=colors,
            **kargs
        )
        
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(REGIONS)
    ax.set_xlabel('Regions')
    ax.set_ylabel('GDP')
    ax.legend(legends, labels=INDUSTRIES_eng)
    return industries_gdp


# 提取每个区域，特定年份的人口数量
def plot_pop(ax, colors, start_yr, end_yr, **kargs):
    # 加载阈值为 0.05的数据，即与黄河流域相交面积大于全市总面积 5% 的所有市
    city_yr = pd.read_csv('../data/perfectures/yr/perfectures_in_YR_with_threshold_0.05.csv')

    # 包含人口的列
    population = ['Urban population', 'Rural population']
    pop = city_yr[['Year', 'Region'] + population]
    for i, region in enumerate(REGIONS):
        use_data = pop.groupby(['Region', 'Year']).sum().loc[region]  # 关注区域的年人口变化
        legends = plot_bar_by_category(
            ax=ax,
            data=use_data,
            categories=population,
            position=i,
            start_yr=start_yr,
            end_yr=end_yr,
            colors=colors,
            **kargs
        )
    ax.legend(labels=['Urban', 'Rural'])
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(REGIONS)
    ax.set_xlabel('Regions')
    ax.set_ylabel('Population')
    return legends


# 提取每个区域，特定年份的人口数量
def plot_wu(ax, colors, start_yr, end_yr, **kargs):
    # 包含人口的列
    city_yr = pd.read_csv('../data/perfectures/yr/perfectures_in_YR_with_threshold_0.05.csv')
    water_use = city_yr[['Year', 'Region', 'Total water use']]
    legends = []
    for i, region in enumerate(REGIONS):
        use_data = water_use.groupby(['Region', 'Year']).sum().loc[region]  # 关注区域的年人口变化
        mean, std = extract_data_by_yr(use_data, start_yr, end_yr)     
        bar = ax.bar(
                x=i,
                height=mean,
                yerr=std,
                color=colors[i],
                **kargs
            )
        legends.append(bar)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(REGIONS)
    ax.set_xlabel('Regions')
    ax.set_ylabel('Total water use')


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
    """计算从此区域及其上游区域，合计的【水资源】消耗量"""
    city_yr = pd.read_csv('../data/perfectures/yr/perfectures_in_YR_with_threshold_0.05.csv')
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


def plot_water(ax, start_yr, end_yr, colors):
    legends = []  # 图例
    for i, region in enumerate(REGIONS):  # 循环每个区域
        measured_runoff = get_runoff_difference(region)  # 获取这个区域的径流量
        wu = get_consumptions(region)  # 获取该区域的消耗量
        c_ser = get_surface_groundwater_coefficient(region)
        c = extract_data_by_yr(c_ser, start_yr, end_yr)[0]
        natual_runoff = measured_runoff + wu * (1-c)
        mean, std = extract_data_by_yr(natual_runoff, start_yr, end_yr)
        bar = ax.bar(
            x=i,
            height=mean,
            yerr=std,
            color=colors[i]
        )
        legends.append(bar)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(REGIONS)
    ax.set_xlabel('Regions')
    ax.set_ylabel('Natural yield')