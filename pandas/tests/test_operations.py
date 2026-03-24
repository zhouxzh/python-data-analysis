"""
测试数据操作功能
"""
import pandas as pd
import pytest

def test_filter():
    df = pd.read_csv('data/students.csv')
    result = df[df['成绩'] > 85]
    assert len(result) > 0
    assert all(result['成绩'] > 85)

def test_groupby():
    df = pd.read_csv('data/sales.csv')
    grouped = df.groupby('产品')['销量'].sum()
    assert len(grouped) > 0
