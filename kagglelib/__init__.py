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
from .kaggle import REVERSE_SALARY_THRESHOLDS
from .kaggle import load_orig_kaggle_df
from .kaggle import load_questions_df
from .kaggle import get_threshold
from .kaggle import load_thresholds_df
from .kaggle import load_udf
from .kaggle import filter_df
from .kaggle import load_role_df
from .kaggle import keep_demo_cols
from .kaggle import load_salary_medians_df
from .kaggle import load_median_salary_per_income_group_per_XP_level_df
from .kaggle import load_median_salary_comparison_df
from .kaggle import fix_age_bin_distribution
from .kaggle import get_age_bin_distribution_comparison
from .kaggle import calc_avg_age_distribution
from .paths import DATA
from .plots import hv_plot_value_count_comparison
from .plots import sns_plot_value_count_comparison
from .plots import PALETTE_INCOME_GROUP
from .plots import PALETTE_ORIGINAL_VS_FILTERED
from .plots import PALETTE_USA_VS_ROW
from .plots import sns_plot_salary_medians
from .plots import sns_plot_age_distribution
from .plots import sns_plot_global_salary_distribution_comparison
from .plots import sns_plot_salary_pde_comparison_per_income_group
#from .plots import sns_plot_salary_pde_comparison_per_income_group2
from .plots import sns_plot_salary_pde_comparison_per_role
from .third_party import load_eurostat_df
from .third_party import get_usd_eur_rate
from .third_party import load_world_bank_groups
from .third_party import load_oecd_df
from .third_party import load_numbeo_df
from .third_party import load_ilo_df
from .third_party import load_mean_salary_comparison_df
from .utils import get_value_count_df
from .utils import stack_value_count_df
from .utils import get_value_count_comparison
from .utils import stack_value_count_comparison
from .utils import stack_dataframe
from .utils import get_stacked_value_count_comparison
from .utils import get_complimentary_datasets
