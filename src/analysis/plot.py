import os
from datetime import datetime
import plotly.express as px
import pandas as pd


def plot_dataframe(df: pd.DataFrame, title: str, show_plot=True, save=True, save_target=None, mode='both', facet_col="mode", facet_row="game_size"):
    # colors = ["#FFA500", "#6495ED"]
    if mode in ['train', 'test']:
        mask = df['mode'] == mode
        df = df[mask]
        # colors = ["#FFA500", "#6495ED"] if mode == 'test' else ["#FF2D00", "#3556BB"]

    figure = px.line(df, x='epoch', y='acc', facet_col=facet_col,
                     facet_row=facet_row, color='plotname',
                     facet_col_spacing=0.01, facet_row_spacing=0.01)

    # Add a horizontal line to each subplot representing the chance level
    for i, facet_row_value in enumerate(reversed(df[facet_row].unique()), start=1):
        for j, facet_col_value in enumerate(df[facet_col].unique(), start=1):
            chance_level = 1 / facet_row_value

            figure.add_shape(type='line',
                             x0=figure.data[0]['x'].min(), x1=figure.data[0]['x'].max(),
                             y0=chance_level, y1=chance_level,
                             line=dict(color='darkgray', width=5, dash='dash'),
                             row=i, col=j)

    # Add a dummy legend item for the chance level
    figure.add_trace(dict(x=[None], y=[None], name='Chance level', line=dict(color='darkgray', width=5),
                          mode='lines', showlegend=True, legendgroup='legend', hoverinfo='skip'))

    figure.update_layout(
        yaxis=dict(range=[-0.05, 1.05]),
        showlegend=True,
        font=dict(family="Arial", size=34, color='#000000'),
        width=1500, height=2100,
        margin=dict(l=0, r=10, t=0, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    title=title, font=dict(size=32), itemsizing='trace')
    )

    figure.update_traces(mode='lines+markers', marker=dict(size=4, line=dict(width=1)), line=dict(width=6))

    show_plot and figure.show()

    if save:
        if not save_target:
            os.makedirs("results", exist_ok=True)
            now = datetime.now().strftime("%Y_%d_%m_%H_%M_%S__")
            save_target = f"results/{now}"

        title = title.replace('<br>', '')

        # orca requires external installation, can use pip install kaleido instead
        figure.write_image(f"{save_target}/{str(title)}.png", engine="orca")
