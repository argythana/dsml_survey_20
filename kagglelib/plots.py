import holoviews as hv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from typing import Optional

from .paths import DATA


SMALL_FONT = 12
MEDIUM_FONT = 18
BIG_FONT = 25
HUGE_FONT = 35


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
    if len(df) > 50:
        raise ValueError(f"You probably don't want to create a Bar plot with 50+ bins: {len(df)}")
    if len(df.columns) != 3 or set(df.columns[-2:]).difference(("source", "Number", "Percentage")):
        raise ValueError(f"The df does not seem to be comparing value_counts: {df.columns}")


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


def sns_plot_value_count_comparison(
    df: pd.DataFrame,
    title: Optional[str] = None,
) -> None:
    check_df_is_stacked(df)
    column = df.columns[0]
    if title is None:
        title = column
    with sns.plotting_context("notebook", rc=MPL_RC):
        plot = sns.catplot(
            data=df,
            kind="bar",
            x=df.columns[0],
            y=df.columns[-1],
            hue=df.columns[1],
            palette="dark",
            alpha=0.6,
            height=8,
            aspect=2.0,
            legend=False,
        )
        # plot.despine(left=True)
        plot.ax.legend(loc="best", title="Source")
        plot.ax.set_title(title)
        # plot.set_axis_labels(x_label, y_label)

        # adapted from: https://stackoverflow.com/questions/39444665/add-data-labels-to-seaborn-factor-plot
        for i, bar in enumerate(plot.ax.patches):
            h = bar.get_height()
            plot.ax.text(
                x=bar.get_x() + bar.get_width() / 2,
                y=h + 0.35,
                s=f"{h:.1f}",  # the label
                ha="center",
                va="center",
                # fontweight='bold',
            )
