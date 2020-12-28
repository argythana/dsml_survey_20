from __future__ import annotations

import functools
import math

from typing import Optional
from typing import Union

import holoviews as hv
import pandas as pd
import seaborn as sns

from holoviews import opts as hv_opts

from .kaggle import SALARY_THRESHOLDS
from .kaggle import load_orig_kaggle_df
from .kaggle import load_questions_df
from .kaggle import get_threshold
from .kaggle import load_thresholds_df
from .kaggle import load_udf
from .kaggle import filter_df
from .kaggle import load_role_df
from .kaggle import keep_demo_cols
from .kaggle import load_salary_medians_df
from .paths import DATA
from .plots import hv_plot_value_count_comparison
from .plots import sns_plot_value_count_comparison
from .third_party import load_eurostat_df
from .third_party import get_usd_eur_rate
from .third_party import load_world_bank_groups
from .third_party import load_eurostat_df
from .third_party import load_oecd_df
from .third_party import load_numbeo_df
from .third_party import load_ilo_df
from .third_party import load_mean_salary_comparison_df




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


def stack_value_count_df(df: pd.DataFrame):
    if len(df) > 50:
        raise ValueError(f"You probably don't want to create a Bar plot with 50+ bins: {len(df)}")
    if len(df.columns) != 4 or list(df.columns[-3:]) != ["original", "filtered", "% diff"]:
        raise ValueError(f"The df does not seem to be comparing value_counts: {df.columns}")
    column: str = df.columns[0]
    y_label = "Percentage" if math.isclose(100, df["original"].sum(), abs_tol=0.5) else "Number"
    df = df.drop(columns="% diff").set_index(column).stack().reset_index()
    df.columns = [column, "source", y_label]
    return df
