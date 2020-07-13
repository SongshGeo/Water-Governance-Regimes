# !/usr/bin/env python
# - * - coding: utf-8 - * -
# author: Shuang Song time: 2020/1/14
# Email: SongshGeo@Gmail.com
# project: 

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from constants import UP, MIDDLE, DOWN
import csv


def judge_region(data):
    up_df = pd.DataFrame(columns=data.columns)
    mid_df = pd.DataFrame(columns=data.columns)
    down_df = pd.DataFrame(columns=data.columns)
    for region, group in data.groupby('分区'):
        if region in UP:
            up_df = up_df.append(group)
        elif region in MIDDLE:
            mid_df = mid_df.append(group)
        elif region in DOWN:
            down_df = down_df.append(group)
        else:
            pass
    return [up_df, mid_df, down_df]



# 将dbf文件读取入pandas.DataFrame
def pd_read_dbf(io, usecols=False):
    """从文件系统中读取dbf文件并写入DataFrame
    io: 文件路径，
    usecols: 使用的列"""
    from dbfread import DBF  # 加载dbfreead包才能读取文件
    if io.endswith('.dbf'):
        table = DBF(io)
    else:
        raise Exception("{} is not a dbf file.".format(io.split("/")[-1]))
    df = pd.DataFrame(table)
    if usecols:
        for col in df:
            if col not in usecols:
                df.drop(col, axis=1, inplace=True)
    return df
