from __future__ import annotations

import functools
import math

from typing import Optional
from typing import Union

import holoviews as hv
import pandas as pd
import seaborn as sns

from holoviews import opts as hv_opts

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




# Dictionaries of useful CONSTANTS
SALARY_THRESHOLDS = {
    "$0-999": 1000,
    "1,000-1,999": 2000,
    "2,000-2,999": 3000,
    "3,000-3,999": 4000,
    "4,000-4,999": 5000,
    "5,000-7,499": 7500,
    "7,500-9,999": 10000,
    "10,000-14,999": 15000,
    "15,000-19,999": 20000,
    "20,000-24,999": 25000,
    "25,000-29,999": 30000,
    "30,000-39,999": 40000,
    "40,000-49,999": 50000,
    "50,000-59,999": 60000,
    "60,000-69,999": 70000,
    "70,000-79,999": 80000,
    "80,000-89,999": 90000,
    "90,000-99,999": 100000,
    "100,000-124,999": 125000,
    "125,000-149,999": 150000,
    "150,000-199,999": 200000,
    "200,000-249,999": 250000,
    "250,000-299,999": 300000,
    "300,000-500,000": 500000,
    "> $500,000": 1000000,
}

_KAGGLE_ROLES = set(
    [
        "Business Analyst",
        "Currently not employed",
        "DBA/Database Engineer",
        "Data Analyst",
        "Data Engineer",
        "Data Scientist",
        "Machine Learning Engineer",
        "Other",
        "Product/Project Manager",
        "Research Scientist",
        "Software Engineer",
        "Statistician",
        "Student",
    ]
)

_KAGGLE_RENAMES = {
    "Time from Start to Finish (seconds)": "duration",
    "Q1": "age",
    "Q2": "gender",
    "Q3": "country",
    "Q4": "education",
    "Q5": "role",
    "Q6": "code_exp",
    "Q15": "ml_exp",
    "Q20": "employees",
    "Q21": "team_ds",
    "Q22": "company_ml_use",
    "Q24": "salary",
    "Q25": "spend_ds",
}




@functools.lru_cache(maxsize=1)
def load_orig_kaggle_df() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "kaggle_survey_2020_responses.csv",
        header=0,
        low_memory=False,
    )
    return df


@functools.lru_cache(maxsize=1)
def load_questions_df() -> pd.DataFrame:
    orig = load_orig_kaggle_df()
    questions_df = orig.loc[0].reset_index(drop=True)
    return questions_df


def get_threshold(value: float, offset: int = 1):
    thresholds = list(SALARY_THRESHOLDS.values())
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            break
    index = max(0, i - offset)
    return thresholds[index]


@functools.lru_cache(maxsize=1)
def load_thresholds_df(
    low_salary_percentage: float = 0.33,
    low_salary_high_exp_offset: int = 2,
    high_salary_low_exp_threshold: int = 500000,
) -> pd.DataFrame:
    df = load_mean_salary_comparison_df()
    df = df[["country", "income_group", "avg_salary"]]
    df = df.append(dict(country="Other", avg_salary=3500), ignore_index=True)
    df = df.assign(
        too_low_salary=(low_salary_percentage * df.avg_salary).apply(get_threshold),
        low_salary_high_exp=df.avg_salary.apply(lambda x: get_threshold(x, low_salary_high_exp_offset)),
        high_salary_low_exp=high_salary_low_exp_threshold,
    )
    return df


@functools.lru_cache(maxsize=1)
def load_udf() -> pd.DataFrame:
    orig = load_orig_kaggle_df()

    # The first row is the "questions". Not real data, so drop it.
    df = orig.loc[1:].reset_index(drop=True)

    # Rename columns to something more convenient
    df = df.rename(columns=_KAGGLE_RENAMES)

    # Cast duration to an integer
    df = df.assign(duration=df.duration.apply(int))

    # Align country names to the Official datasets' names
    # There are two different choices for 'Korea' in Kaggle dataset.
    # We assume that both choices refer to the country in the southern part of the Peninsula.
    df.country = df.country.replace(
        {
            "United States of America": "USA",
            "United Kingdom of Great Britain and Northern Ireland": "UK",
            "Iran, Islamic Republic of...": "Iran",
            "Republic of Korea": "Korea, Republic of",
            "South Korea": "Korea, Republic of",
        }
    )

    # Columns about experience have different ranges and different format.
    # Modify format to be similar and DNRY
    # This way, we minimize errors that may be caused by human typing,
    # e.g. in the executive summary p. 10, machine learning experience class from 10-20 years is referenced as 10-15 years
    df.code_exp = df.code_exp.replace({"< 1 year": "< 1", "I have never written code": "0"}).str.replace(" years", "")
    df.ml_exp = df.ml_exp.replace(
        {"Under 1 year": "< 1", "20 or more years": "20+", "I do not use machine learning methods": "0"}
    ).str.replace(" years", "")

    # Refine Company employment size values
    df.employees = (
        df.employees.replace(
            {
                "10,000 or more employees": "10000+",
            }
        )
        .str.replace(" employees", "")
        .str.replace(",", "")
    )

    # create salary upper bound thresholds for comparison operations.
    df = df.assign(salary_threshold=df.salary.map(SALARY_THRESHOLDS))

    # Reformat salary bins by removing symbols and "," from salary ranges.
    df.salary = df.salary.replace(
        {
            "$0-999": "0-999",
            "> $500,000": "500,000-999,999",
            "300,000-500,000": "300,000-499,999",
        }
    ).str.replace(",", "")

    # convert spend_ds ranges to upper bounds (i.e. integers):
    df.spend_ds = df.spend_ds.replace(
        {
            "$0 ($USD)": 0,
            "$1-$99": 100,
            "$100-$999": 1000,
            "$1000-$9,999": 10000,
            "$10,000-$99,999": 100000,
            "$100,000 or more ($USD)": 1000000,
        }
    )

    # Merge df with the threshold values
    thresholds = load_thresholds_df()
    df = pd.merge(df, thresholds, how="inner", on="country")

    return df


def filter_df(df: pd.DataFrame, print_filters=False) -> pd.DataFrame:
    # Remove those who only answered "demographic" questions
    # Q7 is the first non-demographic question
    # While 5 are the "threshold" columns which were appended at the end
    temp_df = df.iloc[:, 7:-5]
    only_answer_demographic = ((temp_df == "None") | temp_df.isnull()).all(axis=1)
    # Basic conditions
    low_exp_bins = ["0", "< 1", "1-2"]
    is_low_exp = df.code_exp.isin(low_exp_bins) & (df.ml_exp.isin(low_exp_bins) | df.ml_exp.isna())
    high_exp_bins = ["10-20", "20+"]
    is_high_exp = df.code_exp.isin(high_exp_bins) | df.ml_exp.isin(high_exp_bins)
    is_too_low_salary = df.salary_threshold <= df.too_low_salary
    # complex conditions
    is_too_young_for_experience = (df.age <= "24") & ((df.code_exp == "20+") | (df.ml_exp == "20+"))
    is_too_young_for_salary = (df.age <= "24") & (df.salary_threshold >= df.high_salary_low_exp)
    is_low_salary_high_exp = is_high_exp & (df.salary_threshold < df.low_salary_high_exp)
    is_high_salary_low_exp = is_low_exp & (df.salary_threshold >= df.high_salary_low_exp)

    if print_filters:
        print(
            f"""
        Too young for experience   : {is_too_young_for_experience.sum()}
        Too young for salary       : {is_too_young_for_salary.sum()}
        Too low salary             : {is_too_low_salary.sum()}
        Too low salary high exp    : {is_low_salary_high_exp.sum()}
        Too high salary low exp    : {is_high_salary_low_exp.sum()}
        only_answered_demographics : {only_answer_demographic.sum()}
        """
        )

    # Create dataframe
    conditions = (
        ~is_too_young_for_experience
        & ~is_too_young_for_salary
        & ~is_too_low_salary
        & ~is_low_salary_high_exp
        & ~is_high_salary_low_exp
        & ~only_answer_demographic
    )
    df = df[conditions]
    return df


def load_role_df(df: pd.DataFrame, role: str) -> pd.DataFrame:
    if role not in _KAGGLE_ROLES:
        raise ValueError(f"Unknown role: {role}")
    df = df[df.role == role]
    return df


def value_counts(series: pd.Series, sort_index: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "counts": series.value_counts().sort_values(),
            "(%)": (series.value_counts(True).sort_values() * 100).round(2),
        }
    )
    if sort_index:
        df = df.sort_index()
    else:
        df = df.sort_values(by="counts", ascending=False)
    df.index.name = series.name
    return df


def keep_demo_cols(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_keep = [col for col in df.columns if not col.startswith("Q")]
    df = df[columns_to_keep]
    return df


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
