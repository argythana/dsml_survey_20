import functools

from typing import Union

import pandas as pd

from .paths import DATA


@functools.lru_cache(maxsize=1)
def load_usd_eur_df() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "ecb_usd_euro_avg_exch_rate_filtered.csv",
        header=None,
        skiprows=5,
        names=["year", "rate"],
        index_col="year",
    )
    return df


@functools.lru_cache(maxsize=1)
def get_usd_eur_rate(year: int) -> Union[int, str, float]:
    df = load_usd_eur_df()
    rate = df.at[year, "rate"]
    return rate


@functools.lru_cache(maxsize=1)
def load_world_bank_groups() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "wb_country_income_groups.csv",
        header=0,
        names=["country", "income_group"],
        index_col="country",
    )
    df.income_group = df.income_group.str.replace(" income", "")
    return df


@functools.lru_cache(maxsize=1)
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


@functools.lru_cache(maxsize=1)
def load_oecd_df() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "oecd_ann_avg_wage_2019.csv",
        header=0,
        names=["country", "oecd"],
        index_col="country",
    )
    df = df.assign(year=2019)
    return df


@functools.lru_cache(maxsize=1)
def load_numbeo_df() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "numbeo.csv",
        header=0,
        names=["country", "numbeo"],
        index_col="country",
    )
    df.numbeo = df.numbeo * 12
    df = df.assign(year=2020)
    return df


@functools.lru_cache(maxsize=1)
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


@functools.lru_cache(maxsize=1)
def load_mean_salary_comparison_df():
    income_group = load_world_bank_groups()
    eurostat = load_eurostat_df()
    oecd = load_oecd_df()
    ilo = load_ilo_df()
    numbeo = load_numbeo_df()
    df = pd.concat([income_group.income_group, eurostat.eurostat, oecd.oecd, ilo.ilo, numbeo.numbeo], axis="columns")
    df.index.name = "country"
    df = df.assign(country_avg_salary=df.bfill(axis=1).iloc[:, 1])
    df = df.reset_index(drop=False)
    return df
