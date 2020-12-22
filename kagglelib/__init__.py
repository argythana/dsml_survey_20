from __future__ import annotations

import pathlib

from typing import List
from typing import Union

import pandas as pd


ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"


__all__: List[str] = [
    "load_usd_eur_df",
    "load_eurostat_df",
    "get_usd_eur_rate",
]


def load_usd_eur_df() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "ecb_usd_euro_avg_exch_rate_filtered.csv",
        header=None,
        skiprows=5,
        names=["year", "rate"],
        index_col="year"
    )
    return df


def get_usd_eur_rate(year: int) -> Union[int, str, float]:
    df = load_usd_eur_df()
    rate = df.at[year, "rate"]
    return rate


def load_eurostat_df() -> pd.DataFrame:
    usd_eur = get_usd_eur_rate(2019)
    df = pd.read_csv(
        DATA / "eurostat_gross_earnings_euros_2019_filtered.csv",
        header=0,
        names=["country", "eurostat"],
        index_col="country",
    )
    df = df.assign(year=2019, eurostat=df.eurostat * usd_eur)
    return df


def load_oecd_df() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / 'oecd_ann_avg_wage_2019.csv',
        header=0,
        names=["country", "oecd"],
        index_col="country",
    )
    df = df.assign(year=2019)
    return df


def load_numbeo_df() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "numbeo.csv",
        header=0,
        names=["country", "numbeo"],
        index_col="country",
    )
    df = df.assign(year=2020)
    return df


def load_ilo_df() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "ilo_mean_nom_wage_usd.csv",
        header=0,
        usecols=[0, 5, 6],
        names=["country", "year", "monthly"],
    )
    df = df.groupby("country").tail(1)
    df = df.assign(ilo=df.monthly * 12)
    df = df.drop(columns="monthly")
    df = df.set_index("country")
    return df


def load_orig_kaggle_df() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "kaggle_survey_2020_responses.csv",
        header=0,
        low_memory=False,
    )
    return df


def load_questions_df() -> pd.DataFrame:
    orig = load_orig_kaggle_df()
    questions_df = orig.loc[0].reset_index(drop=True)
    return questions_df


# Select useful columns for data validity exploration
KAGGLE_VALIDATION_COLS = {
    'Q1': 'age',
    'Q2': 'gender',
    'Q3': 'country',
    'Q4': 'education',
    'Q5': 'role',
    'Q6': 'code_exp',
    'Q15': 'ml_exp',
    'Q20': 'employees',
    'Q21': 'team_ds',
    'Q22': 'company_ml_use',
    'Q25': 'spend_ds',
    'Q24': 'salary',
    'salary_threshold': 'salary_threshold',
}

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


def load_kaggle_df() -> pd.DataFrame:
    orig = load_orig_kaggle_df()

    # The first row is the "questions". Not real data, so drop it.
    df = orig.loc[1:].reset_index(drop=True)

    # Column 1: "Time from Start to Finish (seconds)" contains integers.
    # Let's cast it and rename it to something more convenient
    df = df.rename(columns={'Time from Start to Finish (seconds)': 'duration'})
    df = df.assign(duration=df.duration.apply(int))

    # Reformat country names to align with Official datasets' names
    # There are two different choices for 'South Korea' in Kaggle dataset.
    # Both choices refer to the country in the southern part of the Peninsula.
    df.Q3 = df.Q3.replace({
        'Russia': 'Russian Federation',
        'United States of America': 'United States',
        'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
        'Iran, Islamic Republic of...': 'Iran',
        'Republic of Korea': 'Korea, Republic of',
        'South Korea': 'Korea, Republic of'
    })

    # Columns about experience have different ranges and different format.
    # Modify format to be similar and DNRY
    # This way, we minimize errors that may be caused by human typing,
    # e.g. executive summary p. 10, machine learning experience class from 10-20 years is reference as 10-15 years
    df.Q6 = df.Q6.replace({
        '< 1 year': '< 1',
        'I have never written code': '0'
    }).str.replace(' years', '')
    df.Q15 = df.Q15.replace({
        'Under 1 year': '< 1',
        '20 or more years': '20+',
        'I do not use machine learning methods': '0'
    }).str.replace(' years', '')

    # Refine Company employment size values
    df.Q20 = df.Q20.replace({
        '10,000 or more employees': '10000+',
    }).str.replace(' employees', '').str.replace(',', '')

    # convert spend_ds to numbers:
    df.Q25 = df.Q25.replace({
        '$0 ($USD)': 0,
        '$1-$99': 100,
        '$100-$999': 1000,
        '$1000-$9,999': 10000,
        '$10,000-$99,999': 100000,
        '$100,000 or more ($USD)': 1000000,
    })

    # create salary upper bound thresholds for comparison operations.
    df['salary_threshold'] = df['Q24'].map(SALARY_THRESHOLDS)

    # Reformat salary bins by removing symbols and "," from salary ranges.
    df.Q24 = df.Q24.replace({
        '$0-999': '0-999',
        '> $500,000': '500,000-999,999',
        '300,000-500,000': '300,000-499,999',
    }).str.replace(',', '')

    df = df[KAGGLE_VALIDATION_COLS.keys()]
    df = df.rename(columns=KAGGLE_VALIDATION_COLS)
    return df


def load_mean_salary_comparison_df():
    eurostat = load_eurostat_df()
    oecd = load_oecd_df()
    ilo = load_ilo_df()
    numbeo = load_numbeo_df()
    df = pd.concat([eurostat.eurostat, oecd.oecd, ilo.ilo, numbeo.numbeo], axis="columns")
    df.index.name = "country"
    df = df.assign(external_mean=df.bfill(axis=1).iloc[:, 0])
    df = df.reset_index(drop=False)
    return df
