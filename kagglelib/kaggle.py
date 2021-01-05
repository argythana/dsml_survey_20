import functools
import textwrap

import pandas as pd

from .paths import DATA
from .third_party import load_mean_salary_comparison_df
from .utils import stack_dataframe
from .utils import stack_value_count_df
from .utils import stack_value_count_comparison

from typing import List
from typing import Optional
from typing import Union

YEARS_PER_BIN = {
    "18-21": 4,
    "22-24": 3,
    "25-29": 5,
    "30-34": 5,
    "35-39": 5,
    "40-44": 5,
    "45-49": 5,
    "50-54": 5,
    "55-59": 5,
    "60-69": 10,
    "70+": 10
}

SALARY_THRESHOLDS = {
    "0-999": 1000,
    "1000-1999": 2000,
    "2000-2999": 3000,
    "3000-3999": 4000,
    "4000-4999": 5000,
    "5000-7499": 7500,
    "7500-9999": 10000,
    "10000-14999": 15000,
    "15000-19999": 20000,
    "20000-24999": 25000,
    "25000-29999": 30000,
    "30000-39999": 40000,
    "40000-49999": 50000,
    "50000-59999": 60000,
    "60000-69999": 70000,
    "70000-79999": 80000,
    "80000-89999": 90000,
    "90000-99999": 100000,
    "100000-124999": 125000,
    "125000-149999": 150000,
    "150000-199999": 200000,
    "200000-249999": 250000,
    "250000-299999": 300000,
    "300000-499999": 500000,
    "500000-999999": 1000000,
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
    "0": "1. low XP",
    "0-1": "1. low XP",
    "1-2": "1. low XP",
    "3-5": "2. med XP",
    "5-10": "2. med XP",
    "10-20": "3. high XP",
    "20+": "3. high XP",
}

ML_EXP_LEVELS={
    "0": "1. low XP",
    "0-1": "1. low XP",
    "1-2": "1. low XP",
    "2-3": "1. low XP",
    "3-4": "2. med XP",
    "4-5": "2. med XP",
    "5-10": "2. med XP",
    "10-20": "3. high XP",
    "20+": "3. high XP",
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


def get_threshold(value: float, offset: int):
    thresholds = list(SALARY_THRESHOLDS.values())
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            break
    index = max(0, i - offset)
    return thresholds[index]


@functools.lru_cache(maxsize=1)
def load_thresholds_df(
    low_salary_percentage: float = 0.4,
    threshold_offset: int = 2,
    high_salary_low_exp_threshold: int = 500000,
) -> pd.DataFrame:
    df = load_mean_salary_comparison_df()
    df = df[["country", "income_group", "country_avg_salary"]]
    df = df.append(dict(country="Other", country_avg_salary=3500), ignore_index=True)
    df = df.assign(
        too_low_salary=(low_salary_percentage * df.country_avg_salary).apply(lambda x: get_threshold(x, threshold_offset)),
        low_salary_high_exp=df.country_avg_salary.apply(lambda x: get_threshold(x, threshold_offset)),
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

    df.education = df.education.replace(
        {
            "Some college/university study without earning a bachelor’s degree": "Studies without a degree",
            "No formal education past high school": "High school",
            "I prefer not to answer": "No answer"
        }
    ).str.replace(" degree", "")

    df.gender = df.gender.replace(
        {
            "Prefer to self-describe": "Self-describe",
            "Prefer not to say": "No answer"
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
    # Add code_level and ml_level columns
    df = df.assign(
        code_level=df.code_exp.map(CODE_EXP_LEVELS),
        ml_level=df.ml_exp.map(ML_EXP_LEVELS),
    )

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

    # Reformat salary bins by removing symbols and "," from salary ranges.
    df.salary = df.salary.replace(
        {
            "$0-999": "0-999",
            "> $500,000": "500,000-999,999",
            "300,000-500,000": "300,000-499,999",
        }
    ).str.replace(",", "")

    # create salary upper bound thresholds for comparison operations.
    df = df.assign(salary_threshold=df.salary.map(SALARY_THRESHOLDS))
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

    # Merge df with the threshold values and keep the same index
    thresholds = load_thresholds_df()
    df = pd.merge(df, thresholds, how="left", on="country")
    assert len(df) == 20036, f"The length of df is not 20036: {len(df)}"
    assert list(df.country.tail(3)) == list(orig.Q3.tail(3)), set(df.country.tail(3)) - set(orig.Q3.tail(3))
    assert df.country_avg_salary.isna().sum() == 0, "There are misspelled countries"

    return df


def filter_df(df: pd.DataFrame, print_filters=False) -> pd.DataFrame:
    # Remove participants who only answered "demographic" questions
    # Q7 is the first non-demographic question
    # We use the "original" dataframe instead of `df` because we add a bunch of extra
    # columns in `df` (e.g. `salary_threshold`) and we would need to be updating
    # the index on iloc each time a new column was added.
    orig = load_orig_kaggle_df()
    temp_df = orig.iloc[1:, 7:].reset_index(drop=True)
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
    # Create dataframe
    conditions = (
        ~is_too_young_for_experience
        & ~is_too_young_for_salary
        & ~is_too_low_salary
        & ~is_low_salary_high_exp
        & ~is_high_salary_low_exp
        & ~only_answer_demographic
    )
    # print summary
    if print_filters:
        print(
            textwrap.dedent(
                f"""
                Too young for experience   : {is_too_young_for_experience.sum()}
                Too young for salary       : {is_too_young_for_salary.sum()}
                Too low salary             : {is_too_low_salary.sum()}
                Too low salary high exp    : {is_low_salary_high_exp.sum()}
                Too high salary low exp    : {is_high_salary_low_exp.sum()}
                Only answered demographics : {only_answer_demographic.sum()}
                ---------------------------------
                All conditions combined    : {len(df) - conditions.sum()}
                """
            )
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
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame,
    countries: List[str],
    label1: str = "Unfiltered",
    label2: str = "Filtered",
) -> pd.DataFrame:
    df = pd.DataFrame({
        label1: dataset1[dataset1.country.isin(countries)].groupby("country").salary_threshold.median(),
        label2: dataset2[dataset2.country.isin(countries)].groupby("country").salary_threshold.median(),
    }).reset_index().reindex(columns=["country", label2, label1])
    df = stack_dataframe(df, key_column="country", values_column="salary", order=countries)
    # Some countries, e.g. Russia, have an even number of partcipatnts,
    # Therefore the median is e.g. 22500 while we only have 20000 and 25000 in `SALARY_THRESHODLS`
    # Therefore we round up these values to the next threshold
    nan_labels = ~df.salary.isin(SALARY_THRESHOLDS.values())
    if nan_labels.any():
        df.loc[nan_labels, "salary"] = df.loc[nan_labels, "salary"].apply(lambda v: get_threshold(v, offset=0))
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


def load_median_salary_per_income_group_per_XP_level_df(
    dataset: pd.DataFrame,
    xp_type: str,
    income_group: Optional[str] = None,
) -> None:
    """
    ## Examples

    kglib.load_median_salary_per_income_group_per_XP_level_df(uds, xp_type="code", income_group="3")
    kglib.load_median_salary_per_income_group_per_XP_level_df(uds, xp_type="ml", income_group="3")
    kglib.load_median_salary_per_income_group_per_XP_level_df(uds, xp_type="ml")
    """
    assert xp_type in ("code", "ml"), "xp_type should be in {'code_level', 'ml_level'}, not: %s" % xp_type
    level_variable = "code_level" if xp_type == "code" else "ml_level"
    if income_group:
        dataset = dataset[~dataset.salary.isna() & dataset.income_group.str.startswith(income_group)]
    df = dataset.groupby([level_variable, "income_group"]).salary_threshold.median().reset_index()
    df = df.sort_values(by=[level_variable, "income_group"])
    return df


def load_median_salary_comparison_df(
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame,
    xp_type: str,
    income_group: str,
    label1: str = "filtered",
    label2: str = "unfiltered",
) -> pd.DataFrame:
    """
    ## Examples

        df = kglib.load_median_salary_comparison_df(uds, fds, xp_type="code", income_group="3")
        kglib.sns_plot_value_count_comparison(
            df, height=8, width=18, bar_width=0.35, title_wrap_length=80,
            title="Data Scientists: Median salary per Code XP level in High Income countries Filtered vs Unfiltered datasets"
        )
    """
    level_variable = "code_level" if xp_type == "code" else "ml_level"
    df1 = load_median_salary_per_income_group_per_XP_level_df(dataset=dataset1, xp_type=xp_type, income_group=income_group)
    df2 = load_median_salary_per_income_group_per_XP_level_df(dataset=dataset2, xp_type=xp_type, income_group=income_group)
    df = pd.merge(df1, df2, on=[level_variable, "income_group"])
    df = df.drop(columns="income_group")
    df.columns = ["code_level", label1, label2]
    df = stack_value_count_df(df, "salary_threshold")
    return df


def fix_age_bin_distribution(df: pd.DataFrame, rename_index: bool = True) -> pd.Series:
    age_bins = df.age.value_counts(True).sort_index() * 100
    value = age_bins.at["18-21"]
    age_bins.at["18-21"] = value / 2
    age_bins.at["22-24"] += value / 2
    if rename_index:
        age_bins = age_bins.rename(index={"18-21": "18-19", "22-24": "20-24"})
    return age_bins


def get_age_bin_distribution_comparison(
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame,
    rename_index: bool,
    label1: str = "original",
    label2="filtered",
) -> pd.DataFrame:
    df = pd.DataFrame({
        label1: fix_age_bin_distribution(dataset1, rename_index=rename_index),
        label2: fix_age_bin_distribution(dataset2, rename_index=rename_index),
    })
    df = df.rename_axis("age").reset_index().round(2)
    df = stack_value_count_comparison(df, "participants (%)")
    return df


def calc_avg_age_distribution(df: pd.DataFrame, rename_index: bool = True) -> pd.Series:
    df = df.groupby(["age"]).size()
    df = df.reset_index(name="participants")
    df = df.assign(years_per_bin = df.age.map(YEARS_PER_BIN))
    df = df.assign(avg_participants = df.participants / df.years_per_bin)
    df = df.drop(["participants", "years_per_bin"], axis=1).set_index("age")
    series = pd.Series(df.avg_participants, index=df.index)
    return series
