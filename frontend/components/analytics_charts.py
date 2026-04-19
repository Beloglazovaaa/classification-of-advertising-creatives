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

            try:
                r = int(hex_color.lstrip("#")[0:2], 16)
                g = int(hex_color.lstrip("#")[2:4], 16)
                b = int(hex_color.lstrip("#")[4:6], 16)
                luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                text_color = "white" if luminance < 0.55 else "#333333"
            except Exception:
                text_color = "white"

            display_text = f"{percent:.1f}%" if percent >= 6 else ""

            fig.add_trace(go.Bar(
                y=[topic_translated],
                x=[percent],
                orientation="h",
                marker={
                    "color": hex_color,
                    "line": {"color": "rgba(255,255,255,0.4)", "width": 0.8},
                },
                text=display_text,
                textposition="inside",
                insidetextanchor="middle",
                textfont={"size": 12, "color": text_color, "family": "sans-serif"},
                name=class_name,
                legendgroup=class_name,
                showlegend=(class_name not in added_to_legend),
                hovertemplate=f"<b>{topic_translated}</b><br>{class_name}: {percent:.1f}%<extra></extra>",
            ))
            added_to_legend.add(class_name)

    bar_height = max(52, min(72, 320 // max(num_topics, 1)))

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 15}},
        barmode="stack",
        bargap=0.35,
        yaxis={
            "categoryorder": "array",
            "categoryarray": topics_translated[::-1],
            "tickfont": {"size": 13},
            "title": None,
            "automargin": True,
        },
        xaxis={
            "title": "Доля цвета в тематике (%)",
            "range": [0, 100],
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.07)",
            "zeroline": False,
        },
        height=max(480, 200 + num_topics * bar_height + 160),
        margin={"l": 120, "r": 30, "t": 60, "b": 110},
        legend={
            "title": None,
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 11, "color": "#555"},
            "itemwidth": 30,
            "itemsizing": "constant",
            "traceorder": "normal",
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
        },
        showlegend=True,
        font={"size": 12, "family": "sans-serif"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_traces(marker_line_width=0)
    for trace in fig.data:
        trace.update(legendrank=1)

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
        textposition="outside",
        insidetextorientation="radial",
        textfont={"size": 11},
        domain={"x": [0.12, 1.0], "y": [0.0, 1.0]},
        hole=0.30,
        direction="clockwise",
        sort=True,
        hovertemplate="<b>%{label}</b><br>Доля: %{percent:.1%}<extra></extra>",
    )])

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 15}},
        showlegend=False,
        margin={"t": 60, "b": 60, "l": 80, "r": 80},
        height=500,
    )

    return fig