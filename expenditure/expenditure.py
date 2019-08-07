#!/usr/bin/python

import sys

import pandas as pd


def expenditure_by_month(csv_file, year):
    df = pd.read_csv(csv_file, names=['date', 'statement', 'dr', 'cr', 'bal'], header=0)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df[["dr", "cr"]] = df[["dr", "cr"]].replace(" ", 0).apply(pd.to_numeric)
    df = df[df['date'].dt.year == year]
    exp_df = df[['date', 'dr', 'cr']]
    exp_df = exp_df.groupby(exp_df['date'].dt.month).agg(['sum'])
    exp_df.columns = exp_df.columns.map('_'.join)
    exp_df.insert(0, 'month', exp_df.index)
    exp_json = exp_df.to_json(orient='records')
    return exp_json


if __name__ == "__main__":
    expenditure = expenditure_by_month(csv_file=sys.argv[1], year=int(sys.argv[2]))
    print(expenditure)
