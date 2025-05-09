import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style(notebook_mode="paper"):
    plt.style.use("default")
    FIG_LARGE = (12, 8)
    plt.rcParams['figure.figsize'] = FIG_LARGE
    # %config InlineBackend.figure_format = "retina"

    sns.set(palette="colorblind", color_codes=True, context="talk")

    if notebook_mode.lower() == "paper":
        tick_params = {
            "xtick.top": True,
            "xtick.direction": "in",
            "ytick.right": True,
            "ytick.direction": "in",
            "axes.spine.right": True
        }
        sns.set_style("ticks", tick_params)
        params = {
            "axes.formatter.limits": (-3, 7),
            "xtick.major.size": 10,
            "ytick.major.size": 10,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
        }
        plt.rcParams.update(params)

    elif notebook_mode.lower() == "dark":
        plt.style.use("cyberpunk")

    else:
        plt.style.use("default")
