import matplotlib.pyplot as plt


def heatmap(df, color_map="Blues", axis=None, **kwargs):
    """Styles the dataframe as a heatmap."""
    return df.style.background_gradient(cmap=color_map, axis=axis, **kwargs)

def plot(dataframe, title=None, ax=None, return_ax=False):
    """Plots each column of the dataframe as a line chart in a single axis."""
    
    if ax is None:
        _, ax = plt.subplots(figsize=(16,3))
    dataframe = dataframe.asfreq("d")  # resample to introduce NaNs at missing dates
    for column in dataframe:
        ax.plot(dataframe[column], label=column)
        
    if title is not None:
        ax.set_title(title, size=16)
    ax.set_ylabel('PM', size=20)
    ax.grid()
    ax.set_ylim(0, dataframe.max().max() + 20)
    ax.tick_params(axis='x', pad=10)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_size(16)
    ax.legend(fontsize=16, framealpha=1.0)
    
    if return_ax:
        return ax
