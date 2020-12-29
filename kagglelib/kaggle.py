import functools

import pandas as pd

from .paths import DATA
from .third_party import load_mean_salary_comparison_df
from .utils import stack_value_count_df

from typing import List


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

REVERSE_SALARY_THRESHOLDS = {v: k for (k, v) in SALARY_THRESHOLDS.items()}

SALARY_AGGREGATE_BINS = {
    "$0-999": 5000,
    "1,000-1,999": 5000,
    "2,000-2,999": 5000,
    "3,000-3,999": 5000,
    "4,000-4,999": 5000,
    "5,000-7,499": 10000,
    "7,500-9,999": 10000,
    "10,000-14,999": 15000,
    "15,000-19,999": 20000,
    "20,000-24,999": 25000,
    "25,000-29,999": 30000,
    "30,000-39,999": 40000,
    "40,000-49,999": 50000,
    "50,000-59,999": 60000,
    "60,000-69,999": 80000,
    "70,000-79,999": 80000,
    "80,000-89,999": 100000,
    "90,000-99,999": 100000,
    "100,000-124,999": 125000,
    "125,000-149,999": 150000,
    "150,000-199,999": 200000,
    "200,000-249,999": 300000,
    "250,000-299,999": 300000,
    "300,000-500,000": 1000000,
    "> $500,000": 1000000,
}

CODE_EXP_LEVELS={
    'minimum': ['0', '0-1'],
    'basic': ['2-3', '3-5'],
    'intermediate': ['5-10'],
    'advanced': ['10-20', '20+']
}

ML_EXP_LEVELS={
    'minimum': ['0', '0-1'],
    'basic': ['2-3'],
    'intermediate': ['3-4', '4-5'],
    'advanced': ['5-10', '10-20', '20+']
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
    df.code_exp = df.code_exp.replace({"< 1 years": "0-1", "I have never written code": "0"}).str.replace(" years", "")
    df.ml_exp = df.ml_exp.replace(
        {"Under 1 year": "0-1", "20 or more years": "20+", "I do not use machine learning methods": "0"}
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
    low_exp_bins = ["0", "0-1", "1-2"]
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


def keep_demo_cols(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_keep = [col for col in df.columns if not col.startswith("Q")]
    df = df[columns_to_keep]
    return df


def load_salary_medians_df(
    unfiltered: pd.DataFrame,
    filtered: pd.DataFrame,
    countries: List[str],
    label1: str = "Filtered",
    label2: str = "Unfiltered",
) -> pd.DataFrame:
    df = pd.concat(
        objs=[
            (
                unfiltered[unfiltered.country.isin(countries)]
                .groupby("country")
                .salary_threshold.median()
                .to_frame()
                .rename(columns={"salary_threshold": "unfiltered_threshold"})
            ),
            (
                filtered[filtered.country.isin(countries)]
                .groupby("country")
                .salary_threshold.median()
                .to_frame()
                .rename(columns={"salary_threshold": "filtered_threshold"})
            ),
        ],
        axis="columns",
    )
    df = df.assign(
        unfiltered=df.unfiltered_threshold.map(REVERSE_SALARY_THRESHOLDS),
        filtered=df.filtered_threshold.map(REVERSE_SALARY_THRESHOLDS),
    )
    # melt dataframe and add the respective labels
    df = (
        df.reset_index()[["country", "unfiltered_threshold", "filtered_threshold"]]
        .melt(id_vars="country")
        .rename(columns={"value": "salary"})
        .replace({"unfiltered_threshold": label1, "filtered_threshold": label2})
    )
    df = df.assign(label=df.salary.map(REVERSE_SALARY_THRESHOLDS))
    df = df.sort_values("salary", ascending=True)
    # Some countries, e.g. Russia, have an even number of partcipatnts,
    # Therefore the median is e.g. 22500 while we only have 20000 and 25000 in `SALARY_THRESHODLS`
    # Therefore we round up these values to the next threshold
    nan_labels = df.label.isna()
    if nan_labels.any():
        func = lambda v: get_threshold(v, offset=0)
        df.loc[nan_labels, "salary"] = df.loc[nan_labels, "salary"].apply(func)
        df.loc[nan_labels, "label"] = df.loc[nan_labels, "salary"].map(REVERSE_SALARY_THRESHOLDS)
    return df


def load_participants_per_country_df(original: pd.DataFrame, filtered: pd.DataFrame, min_no_participants: int):
    original_value_count = (original.country.value_counts(True) * 100).rename_axis("country").reset_index(name="original")
    filtered_value_count = (filtered.country.value_counts(True) * 100).rename_axis("country").reset_index(name="filtered")
    countries = original_value_count[original_value_count.original > min_no_participants].country
    df = pd.merge(original_value_count, filtered_value_count, how="left", on="country")
    df = df[df.country.isin(countries)]
    df = df.fillna(0)  # nan should present themselves if a country is "eliminated" in the filtered dataset
    df = stack_value_count_df(df, y_label="No. Participants")
    return df
