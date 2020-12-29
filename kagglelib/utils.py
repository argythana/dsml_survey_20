from typing import Optional

import pandas as pd


def get_value_count_df(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column: str,
    perc: bool = True,
    label1: str = "original",
    label2: str = "filtered",
):
    multiplier = 100 if perc else 1
    vc1 = df1[column].value_counts(perc) * multiplier
    vc2 = df2[column].value_counts(perc) * multiplier
    df = pd.DataFrame(
        {
            label1: (vc1.sort_index()).round(2),
            label2: (vc2.sort_index()).round(2),
            "% diff": ((vc2 - vc1) / vc1 * 100).round(2),
        }
    )
    df = df.reset_index(drop=False).rename(columns={"index": column})
    return df


def stack_value_count_df(df: pd.DataFrame, y_label: Optional[str] = None):
    column: str = df.columns[0]
    if "% diff" in df.columns:
        df = df.drop(columns="% diff")
    df = df.set_index(column).stack().reset_index()
    df.columns = [column, "source", y_label]
    return df
