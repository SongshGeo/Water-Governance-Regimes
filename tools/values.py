#!/usr/bin/env python 3.83
# -*-coding:utf-8 -*-
# @Author  : Shuang Song
# @Contact   : SongshGeo@gmail.com
# GitHub   : https://github.com/SongshGeo
# Research Gate: https://www.researchgate.net/profile/Song_Shuang9


# 所有黄河流域的省份
PROVINCES = [
    '青海', '青海省', 'Qinghai',
    '四川', '四川省', 'Sichuan',
    '甘肃', '甘肃省', 'Gansu',
    '宁夏', '宁夏回族自治区', 'Ningxia',
    '内蒙', '内蒙古', '内蒙古自治区', 'Neimeng',
    '陕西', '陕西省', 'Shanxi', 
    '山西', '山西省', 'Shaanxi',
    '河南', '河南省', 'Henan',
    '山东', '山东省', 'Shandong'
    '河北', '河北省', 'Hebei'
]

REGIONS = ['SR', 'UR', 'MR', 'DR']  # 设置区域


# 所有GDP相关常量
INDUSTRIES = ['第一', '第二', '第三']  # 三个产业
INDUSTRIES_eng = ['Agriculture', 'Industry', 'Services']  # 三个产业对应的英文


period_colors = ['#0889A6', '#F1801F', '#006C43']
region_colors = ["#0077b6", "#e07a5f", "#f2cc8f","#81b29a"]
index_colors = ['#CFA7D1', '#79D6F0', '#E25A53']


import matplotlib
index_colormap = matplotlib.colors.ListedColormap(index_colors, 'indexed')
total_water_use_color = '#D1495B'