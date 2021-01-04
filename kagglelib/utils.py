from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd


def get_value_count_df(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column: str,
    perc: bool = True,
    label1: str = "Original",
    label2: str = "Filtered"
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
    df = df.rename_axis(column).reset_index(drop=False)
    return df


def stack_value_count_df(df: pd.DataFrame, y_label: Optional[str] = None):
    column: str = df.columns[0]
    if "% diff" in df.columns:
        df = df.drop(columns="% diff")
    df = df.set_index(column).stack().reset_index()
    df.columns = [column, "source", y_label]
    return df


def get_value_count_comparison(
    sr1: pd.Series,
    sr2: pd.Series,
    as_percentage: bool,
    label1: str = "Original",
    label2: str = "Filtered",
    order: Optional[List[str]] = None,
):
    multiplier = 100 if as_percentage else 1
    vc1 = sr1.value_counts(as_percentage) * multiplier
    vc2 = sr2.value_counts(as_percentage) * multiplier
    df = pd.DataFrame(
        {
            label1: vc1.sort_index(),
            label2: vc2.sort_index(),
            "rel diff (%)": (vc2 - vc1) / vc1 * 100,
        }
    )
    if as_percentage:
        df = df.round(2)
    if order:
        df = df.reindex(order)
    df = df.rename_axis(sr1.name).reset_index(drop=False)
    return df


def stack_value_count_comparison(df: pd.DataFrame, stack_label: str):
    column: str = df.columns[0]
    df = df.drop(columns=["% diff", "rel diff (%)"], errors="ignore")
    df = df.set_index(column).stack().reset_index()
    df.columns = [column, "source", stack_label]
    return df


def get_stacked_value_count_comparison(
    sr1: pd.Series,
    sr2: pd.Series,
    stack_label: str,
    as_percentage: bool,
    label1: str = "Original",
    label2: str = "Filtered",
    order: Optional[List[str]] = None,
) -> pd.DataFrame:
    value_count_df = get_value_count_comparison(
        sr1=sr1, sr2=sr2, as_percentage=as_percentage, label1=label1, label2=label2, order=order,
    )
    stacked_df = stack_value_count_comparison(value_count_df, stack_label=stack_label)
    return stacked_df


def get_complimentary_datasets(df: pd.DataFrame, filter: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df1 = df[filter]
    df2 = df[~filter]
    return df1, df2
