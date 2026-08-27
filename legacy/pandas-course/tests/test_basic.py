"""
测试数据读取功能
"""
import pandas as pd
import pytest

def test_read_students():
    df = pd.read_csv('data/students.csv')
    assert len(df) == 5
    assert list(df.columns) == ['姓名', '年龄', '成绩', '城市']

def test_read_sales():
    df = pd.read_csv('data/sales.csv')
    assert len(df) == 5
    assert '产品' in df.columns
