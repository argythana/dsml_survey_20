import holoviews as hv
import matplotlib as mpl
import matplotlib.pyplot as plt
import natsort
import seaborn as sns
import pandas as pd

from textwrap import wrap

from typing import Any
from typing import Dict
from typing import Optional

from .paths import DATA
from .kaggle import REVERSE_SALARY_THRESHOLDS


SMALL_FONT = 12
MEDIUM_FONT = 14
BIG_FONT = 18
HUGE_FONT = 26


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


def hv_plot_value_count_comparison(
    df: pd.DataFrame,
    title: Optional[str] = None,
) -> hv.Layout:
    check_df_is_stacked(df)
    column = df.columns[0]
    if title is None:
        title = column
    source1, source2 = df["source"].unique()
    # Stack dataframe for Bars plot
    plot = hv.Bars(data=df, kdims=[column, "source"], vdims=[df.columns[-1]], label=title)
    plot = plot.opts(
        width=900,
        height=600,
        fontsize=12,
        fontscale=1.0,
        xrotation=90,
        xlabel=f"{source1.capitalize()} VS {source2.capitalize()}",
        show_grid=True,
        show_legend=True,
        show_title=True,
        tools=["hover"],
    )
    layout = plot
    return layout


def get_mpl_rc(rc: Dict[str, Any]) -> Dict[str, Any]:
    mpl_rc = MPL_RC.copy()
    if rc is not None:
        mpl_rc.update(**rc)
    return mpl_rc


# adapted from: https://stackoverflow.com/questions/39444665/add-data-labels-to-seaborn-factor-plot
def _annotate_bar(bar, ax, fmt) -> None:
    h = bar.get_height()
    w = bar.get_width()
    x = bar.get_x()
    ax.annotate(
        text=fmt.format(h),
        xy=(x + w / 2, h),
        xycoords="data",
        ha='center',
        va='center_baseline',
        # offset text 8pts to the top
        xytext=(0, 8),
        textcoords="offset points"
    )


def _annotate_horizontal_bar(bar, ax, fmt) -> None:
    h = bar.get_height()
    w = bar.get_width()
    y = bar.get_y()
    ax.annotate(
        text=fmt.format(w),
        xy=(w, y + h / 2),
        xycoords="data",
        ha='left',
        va='center_baseline',
        # offset text 8pts to the top
        xytext=(3, 0),
        textcoords="offset points"
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
    legend_location: str = "best",
    x_ticklabels_rotation: int = 0,
    bar_width: Optional[float] = None,
) -> None:
    if orientation not in {"horizontal", "vertical", "h", "v"}:
        raise ValueError(f"Orientation must be one of {'horizontal', 'vertical'}, not: {orientation}")
    check_df_is_stacked(df)
    if fmt is None:
        fmt = "{:.1f}" if df.dtypes[-1] == 'float64' else "{:.0f}"
    if title is None:
        title = df.columns[0]
    if orientation in {"horizontal", "h"}:
        x = df.columns[-1]
        y = df.columns[0]
        annotate_func = _annotate_horizontal_bar
        order = natsort.natsorted(df[y].unique())
    else:
        x = df.columns[0]
        y = df.columns[-1]
        annotate_func = _annotate_bar
        order = natsort.natsorted(df[x].unique())
    with sns.plotting_context("notebook", rc=get_mpl_rc(rc)):
        if ax is None:
            fig, ax = plt.subplots(figsize=(width, height))
        sns.barplot(
            data=df,
            ax=ax,
            x=x,
            y=y,
            hue=df.columns[1],
            order=order if order_by_labels else None,
            palette="dark",
            alpha=0.6,
        )
        # Remove Labels from X and Y axes (we should have the relevant info on the title)
        ax.set_xlabel('')
        ax.set_ylabel('')
        if x_ticklabels_rotation != 0:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=x_ticklabels_rotation)
        ax.legend(loc=legend_location, title="Source")
        ax.set_title(title)
        for bar in ax.patches:
            annotate_func(bar, ax, fmt)
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
        )
        ticks = sorted(df.salary.unique(), reverse=True)
        #plot.ax.xticks(ticks, rotation="vertical")
        plt.xticks(ticks, rotation="vertical")
        #plt.xlim((0, 170000))
        x_labels = [REVERSE_SALARY_THRESHOLDS[sal] for sal in ticks]
        #wrapped_x_labels = ['\n'.join(wrap(l, 7)) for l in x_labels]
        wrapped_x_labels = [label.replace("-", "-\n") for label in x_labels]
        plot.ax.xaxis.set_ticklabels(wrapped_x_labels)
        #plot.ax.xaxis.set_ticklabels([REVERSE_SALARY_THRESHOLDS[sal] for sal in ticks])
        plot.ax.grid(axis="x")
        #plot.despine()
        plot.ax.set_axisbelow(True)
        plot.ax.set_box_aspect(12/len(plot.ax.patches))
        plot.ax.legend(loc="center left", title="", bbox_to_anchor=(1.04,0.5))
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
        plt.setp(plot.ax.get_xticklabels(), rotation=30, horizontalalignment='center')
        plt.xticks(fontsize=12)
