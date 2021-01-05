import functools
from textwrap import wrap
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.neighbors

from importlib_metadata import version
from matplotlib.transforms import Bbox

from .paths import DATA
from .kaggle import SALARY_THRESHOLDS
from .kaggle import fix_age_bin_distribution
from .kaggle import calc_avg_age_distribution

PALETTE_INCOME_GROUP = sns.cubehelix_palette(10, rot=-.25, light=.7)
PALETTE_USA_VS_ROW = [sns.desaturate("green", 0.75), "peru"]
PALETTE_ORIGINAL_VS_FILTERED = [sns.desaturate("darkred", 0.90), "darkblue"]


SMALL_FONT = 12
MEDIUM_FONT = 13
BIG_FONT = 14
HUGE_FONT = 18


MPL_RC = {
    "font.size": SMALL_FONT,
    "axes.labelsize": BIG_FONT,
    "axes.titlesize": HUGE_FONT,
    "legend.fontsize": MEDIUM_FONT,
    "legend.title_fontsize": BIG_FONT,
    "xtick.labelsize": MEDIUM_FONT,
    "ytick.labelsize": MEDIUM_FONT,
}


def check_df_is_stacked(df: pd.DataFrame) -> None:
    if len(df.columns) < 3:
        raise ValueError(f"The stacked dataframes need at least 3 columns: {df.columns}")
    if len(df) > 50:
        raise ValueError(f"You probably don't want to create a Bar plot with 50+ bins: {len(df)}")


def get_mpl_rc(rc: Dict[str, Any]) -> Dict[str, Any]:
    mpl_rc = MPL_RC.copy()
    if rc is not None:
        mpl_rc.update(**rc)
    return mpl_rc


def mpl_annotate(ax: mpl.axes.Axes, text: str, **kwargs) -> None:
    """
    Wrapper around ax.annotate() that uses the correct arguments regardless of the matplotlib version

    Matplotlib 3.0 changed the ax.annotate() function definition from:

    ``` diff
    - Axes.annotate(self, s, xy, *args, **kwargs)[source]
    + Axes.annotate(self, text, xy, *args, **kwargs)[source]
    ```

    In Matplotlib 3.0.2 the change was reverted.
    In Matplotlib 3.3 the change was reintroduced .

    ## Relevant links

    - https://matplotlib.org/3.0.0/api/_as_gen/matplotlib.axes.Axes.annotate.html
    - https://matplotlib.org/3.0.2/api/_as_gen/matplotlib.axes.Axes.annotate.html
    - https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.axes.Axes.annotate.html
    - https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.axes.Axes.annotate.html
    - https://github.com/matplotlib/matplotlib/issues/12325/
    - https://github.com/matplotlib/matplotlib/pull/12383
    """
    matplotlib_version = version("matplotlib")
    if matplotlib_version > "3.3":
        kwargs.update({"text": text})
    elif matplotlib_version <= "3.0.1":
        kwargs.update({"text": text})
    else:
        kwargs.update({"s": text})
    ax.annotate(**kwargs)


# adapted from: https://stackoverflow.com/questions/39444665/add-data-labels-to-seaborn-factor-plot
def _annotate_vertical_bar(bar, ax, fmt, annotation_mapping: Optional[Dict[Any, str]] = None):
    h = bar.get_height()
    w = bar.get_width()
    x = bar.get_x()
    if annotation_mapping:
        text = annotation_mapping[h]
    else:
        text = fmt.format(h)
    mpl_annotate(
        ax=ax,
        text=text,
        xy=(x + w / 2, h),
        xycoords="data",
        ha='center',
        va='center_baseline',
        # offset text 8pts to the top
        xytext=(0, 8),
        textcoords="offset points",
        fontweight="bold",
    )


def get_text_width(text: str) -> float:
    ax = plt.gca()
    renderer = ax.figure.canvas.get_renderer()
    text_artist = mpl.text.Text(text=text, figure=ax.figure)
    bbox = text_artist.get_window_extent(renderer=renderer)
    # transform bounding box to data coordinates
    bbox = Bbox(ax.transData.inverted().transform(bbox))
    return bbox.width


def _annotate_horizontal_bar(bar, ax, fmt, annotation_mapping: Optional[Dict[Any, str]] = None) -> None:
    offset = 3  # pts
    h = bar.get_height()
    w = bar.get_width()
    y = bar.get_y()
    if annotation_mapping:
        text = annotation_mapping[w]
    else:
        text = fmt.format(w)
    annotation_width = get_text_width(text)
    if 1.1 * (annotation_width + offset * SMALL_FONT / 72) <= w:
        # annotation is short enough, put it inside the bar
        ha = "right"
        xytext = (-offset, 0)
        color = "white"
    else:
        # annotation too long, put it outside of the bar
        ha = "left"
        xytext = (offset, 0)
        color = "black"
    mpl_annotate(
        ax=ax,
        text=text,
        xy=(w, y + h / 2),
        xycoords="data",
        ha=ha,
        va='center',
        # offset text to the left or right
        xytext=xytext,
        textcoords="offset points",
        fontweight="bold",
        color=color,
    )


def _set_bar_width(bar, width: float) -> None:
    diff = bar.get_width() - width
    bar.set_width(width)  # we change the bar width
    bar.set_x(bar.get_x() + diff / 2)  # we recenter the bar


def sns_plot_value_count_comparison(
    df: pd.DataFrame,
    width: float,
    height: float,
    ax: Optional[mpl.axes.Axes] = None,
    title: Optional[str] = None,
    order_by_labels: bool = True,
    fmt: Optional[str] = None,
    rc: Optional[Dict[str, Any]] = None,
    orientation: str = "vertical",
    legend_location: Optional[str] = "best",
    x_ticklabels_rotation: int = 0,
    bar_width: Optional[float] = None,
    title_wrap_length: Optional[int] = None,
    palette: sns.palettes._ColorPalette  = PALETTE_ORIGINAL_VS_FILTERED,
    annotation_mapping: Optional[Dict[Any, str]] = None,
) -> None:
    if orientation not in {"horizontal", "vertical", "h", "v"}:
        raise ValueError(f"Orientation must be one of {'horizontal', 'vertical'}, not: {orientation}")
    check_df_is_stacked(df)
    if fmt is None:
        fmt = "{:.1f}" if df.dtypes[-1] == 'float64' else "{:.0f}"
    if title is None:
        title = df.columns[0]
    if title_wrap_length:
        title = "\n".join(wrap(title, title_wrap_length))
    if orientation in {"horizontal", "h"}:
        x = df.columns[-1]
        y = df.columns[0]
        annotate_func = _annotate_horizontal_bar
        order = natsort.natsorted(df[y].unique())
    else:
        x = df.columns[0]
        y = df.columns[-1]
        annotate_func = _annotate_vertical_bar
        order = natsort.natsorted(df[x].unique())
    with sns.plotting_context("notebook", rc=get_mpl_rc(rc)):
        sns.set_style("dark", {'axes.linewidth': 0.5})
        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height))
            sns.barplot(
            data=df,
            ax=ax,
            x=x,
            y=y,
            hue=df.columns[1],
            order=order if order_by_labels else None,
            palette=palette,
            alpha=0.6,
        )
        if orientation in {"horizontal", "h"}:
            sns.despine(bottom=True)
            ax.tick_params(left=False, bottom=False)
            ax.xaxis.set_ticklabels("")
        else:
            sns.despine(left=True)
            ax.tick_params(left=False, bottom=False)
            ax.yaxis.set_ticklabels("")
        # Remove Labels from X and Y axes (we should have the relevant info on the title)
        ax.set_xlabel('')
        ax.set_ylabel('')

        if x_ticklabels_rotation != 0:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=x_ticklabels_rotation)
        if legend_location is None:
            ax.get_legend().remove()
        else:
            ax.legend(loc=legend_location, title="")
        ax.set_title(title)
        for bar in ax.patches:
            annotate_func(bar=bar, ax=ax, fmt=fmt, annotation_mapping=annotation_mapping)
            if bar_width:
                _set_bar_width(bar, width=bar_width)


def sns_plot_salary_medians(
    df: pd.DataFrame, title: Optional[str] = None, rc: Optional[Dict[str, Any]] = None
) -> None:
    with sns.plotting_context("notebook", rc=get_mpl_rc(rc)):
        plot = sns.catplot(
            data=df,
            kind="bar",
            x="salary",
            y="country",
            hue="variable",
            orient="h",
            palette="dark",
            alpha=0.8,
            height=8,
            aspect=2.8,
            legend=False,
            order = reversed(df.country.unique()),
            hue_order=["Filtered", "Unfiltered"]
        )

        #ticks = sorted(df.salary.unique(), reverse=True)
        # plot.ax.xticks(ticks, rotation="vertical")
        #plt.xticks(ticks, rotation="vertical")
        #plt.xlim((0, 150000))
        #x_labels = [REVERSE_SALARY_THRESHOLDS[sal] for sal in ticks]
        #wrapped_x_labels = ['\n'.join(wrap(l, 7)) for l in x_labels]
        #wrapped_x_labels = [label.replace("-", "-\n") for label in x_labels]
        #plot.ax.xaxis.set_ticklabels(wrapped_x_labels)
        #plot.ax.xaxis.set_ticklabels([REVERSE_SALARY_THRESHOLDS[sal] for sal in ticks])
        plot.ax.xaxis.set_ticklabels("")
        plot.ax.tick_params(left=False, bottom=False)
        #plot.ax.grid(axis="x")
        plot.despine(bottom=True)
        # Remove Labels from X and Y axes (we should have the relevant info on the title)
        plot.ax.set_xlabel('')
        plot.ax.set_ylabel('')
        # plot.set(xticklabels=[])
        #plot.ax.set_axisbelow(True)
        plot.ax.set_box_aspect(12/len(plot.ax.patches))
        plot.ax.legend(loc="best", title="")
        plot.ax.set_title(title)

        for i, bar in enumerate(plot.ax.patches):
            h = bar.get_height()
            w = bar.get_width()
            y = bar.get_y()
            plot.ax.annotate(
                text=f"${w:.0f}",
                xy=(w, y + h / 2),
                xycoords="data",
                ha='left',
                #va='center',
                va='center_baseline',
                # offset text 4pts to the left
                xytext=(4, 0),
                textcoords="offset points"
            )


def sns_plot_age_distribution(
    df: pd.DataFrame,
    width: float = 14,
    height: float = 10,
    title: str = "Age distribution",
    fmt: str = "{:.1f}",
    rc: Optional[Dict[str, Any]] = None,
    orientation: str = "vertical",
    bar_width: Optional[float] = None,
    title_wrap_length: Optional[int] = None,
) -> None:
    if title_wrap_length:
        title = "\n".join(wrap(title, title_wrap_length))
    default_distribution = (df.age.value_counts(True) * 100).sort_index().round(2)
    proposed_distribution = fix_age_bin_distribution(df, rename_index=True)
    avg_bin_distribution = calc_avg_age_distribution(df, rename_index=True)
    with sns.plotting_context("notebook", rc=get_mpl_rc(rc)):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(width, height))
        color = sns.desaturate("darkred", 0.55)
        sns.barplot(
            x=default_distribution.index,
            y=default_distribution,
            ax=ax1,
            color=color,
        )
        sns.barplot(
            x=proposed_distribution.index,
            y=proposed_distribution,
            ax=ax2,
            color=color,
        )
        sns.barplot(
            x=avg_bin_distribution.index,
            y=avg_bin_distribution,
            ax=ax3,
            color=color,
        )
        ax1.yaxis.set_ticklabels("")
        ax2.yaxis.set_ticklabels("")
        ax3.yaxis.set_ticklabels("")
        sns.despine(ax=ax1, left=True, bottom=True)
        sns.despine(ax=ax2, left=True, bottom=True)
        sns.despine(ax=ax3, left=True, bottom=True)
        ax1.tick_params(left=False, bottom=False)
        ax2.tick_params(left=False, bottom=False)
        ax3.tick_params(left=False, bottom=False)
        ax1.set_title(title)
        ax1.set_ylabel("Default, %")
        ax2.set_ylabel("Adjusted, %")
        ax3.set_ylabel("Average, N")
        for ax in (ax1, ax2, ax3):
            ax1.set_ylim((0, 32))
            ax2.set_ylim((0, 32))
            ax3.set_ylim((0, 1150))
            ax.set_xlabel('')
            #ax.set_ylabel('')
            for bar in ax.patches:
                _annotate_vertical_bar(bar, ax, fmt)
                if bar_width:
                    _set_bar_width(bar, width=bar_width)
        for ax in (ax1, ax2, ax3):
            for i, bar in enumerate(ax.patches):
                if i >= 2:
                    break
                bar.set_color("darkcyan")


def sns_plot_global_salary_distribution_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    width: float,
    height: float,
    x1_limit: Optional[Tuple[float, float]] = (0, 19),
    x2_limit: Optional[Tuple[float, float]] = (0, 19),
    title: str = "Salary Distribution, $",
    fmt: str = "{:.1f}",
    rc: Optional[Dict[str, Any]] = None,
    orientation: str = "vertical",
    bar_width: Optional[float] = None,
    title_wrap_length: Optional[int] = None,
    label1: str = "Unfiltered",
    label2: str = "Filtered",
) -> None:
    if title_wrap_length:
        title = "\n".join(wrap(title, title_wrap_length))
    vc1 = (df1.salary.value_counts(True) * 100).round(2).sort_index().reset_index().rename(columns={"salary": "percentage", "index": "salary"})
    vc2 = (df2.salary.value_counts(True) * 100).round(2).sort_index().reset_index().rename(columns={"salary": "percentage", "index": "salary"})
    order = natsort.natsorted(vc1.salary.unique(), reverse=True)

    with sns.plotting_context("notebook", rc=get_mpl_rc(rc)):
        with sns.axes_style("dark", {'axes.linewidth': 0.5}):
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(width, height), sharey=True, squeeze=True)
            sns.barplot(
                x=vc1.percentage,
                y=vc1.salary,
                ax=ax1,
                order=order,
                palette=[sns.desaturate("red", 0.4)]
            )
            sns.barplot(
                x=vc2.percentage,
                y=vc2.salary,
                ax=ax2,
                order=order,
                palette=[sns.desaturate("cornflowerblue", 0.75)]
            )

            ax1.set_title(label1)
            ax2.set_title(label2)
            ax1.xaxis.set_ticklabels("")
            ax2.xaxis.set_ticklabels("")
            #ax1.set_ylabel("Salary ($)")
            ax1.set_ylabel("")
            ax2.set_ylabel("")
            ax1.set_xlabel("")
            ax2.set_xlabel("")
            ax1.set_xlim((x1_limit))
            ax2.set_xlim((x2_limit))
            ax1.tick_params(left=False, bottom=False)
            ax2.tick_params(left=False, bottom=False)
            ax2.yaxis.set_tick_params(labeltop='on')
            fig.suptitle(title, size=HUGE_FONT)

            plt.tight_layout()
            for ax in (ax1, ax2):
                for i, bar in enumerate(ax.patches):
                    _annotate_horizontal_bar(bar, ax, fmt)
                    if bar_width:
                        _set_bar_width(bar, width=bar_width)


# Define and use a simple function to label the plot in axes coordinates
def _label(x, color, label):
    ax = plt.gca()
    ax.text(
        x=0,
        y=.5,
        s=label,
        fontweight="bold",
        fontsize=BIG_FONT,
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes
    )


def sns_plot_salary_pde_comparison_per_income_group(
    dataset: pd.DataFrame,
    width: float = 18,
    height: float = 10,
    title: str = "Salary PDE per WB income groups (log scale)",
    title_wrap_length: Optional[int] = None,
    bandwidth_adjust: Union[Union[int, float], Tuple[float, float, float, float, float]] = (0.8, 0.6, 0.5, 0.5, 0.5),
    log_scale: bool = True,
    rc: Optional[Dict[str, Any]] = None,
    palette: sns.palettes._ColorPalette = PALETTE_INCOME_GROUP,
) -> None:
    if title_wrap_length:
        title = "\n".join(wrap(title, title_wrap_length))
    if isinstance(bandwidth_adjust, (int, float)):
        bandwidth_adjust = [bandwidth_adjust] *  5
    dataset = dataset[~dataset.salary.isna() & ~(dataset.country == "Other")]
    # global_ = dataset.salary.reset_index(drop=True)
    india = dataset[(dataset.country == "India")].salary_threshold.reset_index(drop=True).rename("India")
    lower_middle = dataset[dataset.income_group.str.startswith("1") & (dataset.country != "India")].salary_threshold.reset_index(drop=True).rename("Lower Middle")
    upper_middle = dataset[dataset.income_group.str.startswith("2")].salary_threshold.reset_index(drop=True).rename("Upper Middle")
    high = dataset[dataset.income_group.str.startswith("3") & (dataset.country != "USA")].salary_threshold.reset_index(drop=True).rename("High")
    usa = dataset[(dataset.country == "USA")].salary_threshold.reset_index(drop=True).rename("USA")
    series = (usa, high, upper_middle, india, lower_middle)
    with sns.plotting_context("notebook", rc=get_mpl_rc(rc)):
        sns.set_style("dark")
        fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(width, height))
        kdeplot_common = functools.partial(
            sns.kdeplot,
            log_scale=log_scale,
            clip_on=False,
            common_norm=False,
            palette=palette,
        )
        kdeplot = functools.partial(
            kdeplot_common,
            fill=True,
            alpha=1,
            linewidth=1.5,
        )
        kdeplot_line = functools.partial(
            kdeplot_common,
            color="w",
            linewidth=2.5,
        )
        for ax, sr, bw_adjust in zip(axes, series, bandwidth_adjust):
            kdeplot(ax=ax, data=sr, bw_adjust=bw_adjust)
            kdeplot_line(ax=ax, data=sr, bw_adjust=bw_adjust)
            ax.set_ylabel(sr.name, rotation=0, ha="right", va="center_baseline")
            ax.yaxis.set_ticklabels("")
            sns.despine(ax=ax, left=True, bottom=True)
            ax.tick_params(left=False, bottom=False)
            ax.xaxis.grid(True)
        ax.set_xlabel("")
        ax.xaxis.set_ticklabels("")
        fig.suptitle(title, size=HUGE_FONT)
        plt.tight_layout()


def sns_plot_pde_comparison(
    series: Union[pd.Series, List[pd.Series]],
    width: float = 18,
    height: float = 14,
    title: str = "Salary PDE per role (log scale)",
    title_wrap_length: Optional[int] = None,
    bandwidth: float = 10,
    log_scale: bool = True,
    rc: Optional[Dict[str, Any]] = None,
    palette: sns.palettes._ColorPalette = PALETTE_INCOME_GROUP,
) -> None:
    if title_wrap_length:
        title = "\n".join(wrap(title, title_wrap_length))
    if isinstance(series, pd.Series):
         series = [series]
    if not isinstance(bandwidth, (list, tuple)):
        bandwidth = [bandwidth] * len(series)
    with sns.plotting_context("notebook", rc=get_mpl_rc(rc)):
        sns.set_style("dark")
        fig, axes = plt.subplots(nrows=len(series), ncols=1, sharex=True, sharey=True, figsize=(width, height))
        for (sr, ax, bw) in zip(series, axes, bandwidth):
            x_d = np.array(sorted(SALARY_THRESHOLDS.values()))
            kde = sklearn.neighbors.KernelDensity(bandwidth=bw, kernel="gaussian")
            kde.fit(sr.values[:, None])
            logprob = kde.score_samples(x_d[:, None])
            ax.plot(x_d, np.exp(logprob), color="w", linewidth=2.5)
            ax.fill_between(x_d, np.exp(logprob), alpha=1, linewidth=1.5)
            ax.set_ylabel(sr.name, rotation=0, ha="right", va="center_baseline")
            sns.despine(ax=ax, left=True, bottom=True)
            ax.tick_params(left=False, bottom=False)
            ax.xaxis.grid(True, which="major")
            ax.yaxis.set_ticklabels("")
        if log_scale:
            ax.set_xscale('log')
        #ax.set_xlim((1, 1000000))
        fig.suptitle(title, size=HUGE_FONT)
        plt.tight_layout()


def sns_plot_salary_pde_comparison_per_role(
    dataset: pd.DataFrame,
    width: float = 18,
    height: float = 14,
    title: str = "Salary PDE per role (log scale)",
    title_wrap_length: Optional[int] = None,
    bandwidth_adjust: float = 0.6,
    log_scale: bool = True,
    rc: Optional[Dict[str, Any]] = None,
    palette: sns.palettes._ColorPalette = PALETTE_INCOME_GROUP,
) -> None:
    if title_wrap_length:
        title = "\n".join(wrap(title, title_wrap_length))
    dataset = dataset[~dataset.salary.isna() & ~(dataset.country == "Other")]
    roles = [
        "Business Analyst",
        "DBA/Database Engineer",
        "Data Analyst",
        "Data Engineer",
        "Data Scientist",
        "Machine Learning Engineer",
        "Research Scientist",
        "Software Engineer",
        "Product/Project Manager",
        "Statistician",
        "Other",
    ]
    series = map(lambda role: dataset[dataset.role == role].salary_threshold.reset_index(drop=True).rename(role), roles)
    with sns.plotting_context("notebook", rc=get_mpl_rc(rc)):
        sns.set_style("dark")
        fig, axes = plt.subplots(nrows=len(roles), ncols=1, sharex=True, figsize=(width, height))
        kdeplot_common = functools.partial(
            sns.kdeplot,
            log_scale=log_scale,
            bw_adjust=bandwidth_adjust,
            clip_on=False,
            common_norm=False,
            palette=palette,
        )
        kdeplot = functools.partial(
            kdeplot_common,
            fill=True,
            alpha=1,
            linewidth=1.5,
        )
        kdeplot_line = functools.partial(
            kdeplot_common,
            color="w",
            linewidth=2.5,
        )
        for ax, sr in zip(axes, series):
            kdeplot(ax=ax, data=sr)
            kdeplot_line(ax=ax, data=sr)
            ax.set_ylabel(sr.name, rotation=0, ha="right", va="center_baseline")
            ax.yaxis.set_ticklabels("")
            sns.despine(ax=ax, left=True, bottom=True)
            ax.tick_params(left=False, bottom=False)
            ax.grid(True)
        ax.set_xlabel("")
        fig.suptitle(title, size=HUGE_FONT)
        plt.tight_layout()
