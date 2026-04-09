import streamlit as st


def color_block_horizontal(colors, title="Цвета", show_percent=True, show_rgb=False):
    if not colors:
        return

    st.markdown(f"**{title}**")

    sorted_colors = sorted(colors, key=lambda x: x.get("percent", 0), reverse=True)
    n_cols = max(1, min(len(sorted_colors), 10))
    cols = st.columns(n_cols, gap="medium")

    for c, col in zip(sorted_colors, cols, strict=False):
        with col:
            st.markdown(
                f"""
                <div style="
                    width: 50px;
                    height: 50px;
                    background-color: {c['hex']};
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                "></div>
                """,
                unsafe_allow_html=True,
            )
            label_parts = [f"<b>{c['hex'].upper()}</b>"]
            if "class_name" in c:
                label_parts.append(f"<medium>{c['class_name']}</medium>")
            if show_percent:
                label_parts.append(f"<medium>{c['percent']:.1f}%</medium>")
            if show_rgb and "rgb" in c:
                label_parts.append(f"<small>RGB({c['rgb'][0]}, {c['rgb'][1]}, {c['rgb'][2]})</small>")

            label_html = "<br>".join(label_parts)
            st.markdown(
                f"<div style='text-align: left; font-size: 13px; line-height: 1.3;'>{label_html}</div>",
                unsafe_allow_html=True,
            )
