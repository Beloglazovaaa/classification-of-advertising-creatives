import plotly.graph_objects as go
from config import COLOR_VISUAL_CLASSES
from config import TOPIC_TRANSLATIONS


def create_topic_color_stacked_bar(topic_color_data, title="Цвета по тематикам"):
    if not topic_color_data:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных", x=0.5, y=0.5, showarrow=False, font={"color": "gray"})
        fig.update_layout(title=title, showlegend=False)
        return fig

    topics_original = list(topic_color_data.keys())
    topics_translated = [TOPIC_TRANSLATIONS.get(t, t) for t in topics_original]
    num_topics = len(topics_original)

    fig = go.Figure()
    added_to_legend = set()

    for topic_orig in reversed(topics_original):
        topic_translated = TOPIC_TRANSLATIONS.get(topic_orig, topic_orig)
        colors = topic_color_data[topic_orig]
        for color_info in colors:
            class_name = color_info["class"]
            percent = color_info["percent"]
            hex_color = color_info["hex"]

            fig.add_trace(go.Bar(
                y=[topic_translated],
                x=[percent],
                orientation="h",
                marker={"color": hex_color, "line": {"color": "#AAAAAA", "width": 0.1}},
                text=f"{percent:.1f}%",
                textposition="inside",
                name=class_name,
                legendgroup=class_name,
                showlegend=(class_name not in added_to_legend),
                hovertemplate=f"<b>{topic_translated}</b><br>{class_name}: {percent:.1f}%<extra></extra>",
            ))
            added_to_legend.add(class_name)

    fig.update_layout(
        title=title,
        barmode="stack",
        yaxis={
            "categoryorder": "array",
            "categoryarray": topics_translated[::-1],
            "tickfont": {"size": 12},
            "title": None,
        },
        xaxis={
            "title": "Доля цвета в тематике (%)",
            "range": [0, 100],
            "showgrid": True,
            "gridcolor": "lightgray",
        },
        height=200 + num_topics * 40,
        margin={"l": 150, "r": 50, "t": 80, "b": 50},
        legend_title="Цвета",
        showlegend=True,
        font={"size": 12},
    )

    return fig


def create_color_pie_chart(class_distribution, title="Распределение цветов"):
    if not class_distribution:
        fig = go.Figure()
        fig.add_annotation(
            text="Нет данных", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font={"size": 14, "color": "gray"},
        )
        fig.update_layout(title=title, showlegend=False)
        return fig

    sorted_items = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colors = []
    for label in labels:
        hex_list = list(COLOR_VISUAL_CLASSES.get(label, {"ffffff"}))
        colors.append(f"#{hex_list[0].upper()}")

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker={"colors": colors},
        textinfo="label+percent",
        texttemplate="%{label}: %{percent:.1%}",
        hovertemplate="<b>%{label}</b><br>Доля: %{percent:.1%}<extra></extra>",
    )])

    fig.update_layout(
        title=title,
        showlegend=False,
        margin={"t": 50, "b": 20, "l": 20, "r": 20},
        height=500,
    )

    return fig
