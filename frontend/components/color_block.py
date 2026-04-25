import streamlit as st


def color_block_horizontal(colors, title="Цвета", show_percent=True, show_rgb=False):
    if not colors:
        return

    st.markdown(
        f"""
        <div style="
            font-family: sans-serif;
            font-size: 13px;
            font-weight: 600;
            color: #999;
            margin-bottom: 12px;
            margin-top: 10px;
            letter-spacing: 0.01em;
        ">{title}</div>
        """,
        unsafe_allow_html=True,
    )

    sorted_colors = sorted(colors, key=lambda x: x.get("percent", 0), reverse=True)
    n_cols = max(1, min(len(sorted_colors), 8))
    cols = st.columns(n_cols, gap="small")

    for c, col in zip(sorted_colors, cols, strict=False):
        with col:
            label_parts = []
            if "class_name" in c:
                label_parts.append(
                    f"<div style='font-family:sans-serif; font-size:12px; font-weight:600; color:#333; "
                    f"white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>"
                    f"{c['class_name']}</div>"
                )
            label_parts.append(
                f"<div style='font-family:monospace; font-size:11px; color:#666;'>"
                f"{c['hex'].upper()}</div>"
            )
            if show_percent:
                label_parts.append(
                    f"<div style='font-family:sans-serif; font-size:11px; color:#999;'>{c['percent']:.1f}%</div>"
                )
            if show_rgb and "rgb" in c:
                label_parts.append(
                    f"<div style='font-family:sans-serif; font-size:10px; color:#bbb;'>"
                    f"RGB({c['rgb'][0]}, {c['rgb'][1]}, {c['rgb'][2]})</div>"
                )

            label_html = "".join(label_parts)

            st.markdown(
                f"""
                <div style="
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 6px;
                    padding: 10px 6px 12px;
                    border: 1px solid #ebebeb;
                    border-radius: 10px;
                    background: #fafafa;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
                    min-width: 0;
                ">
                    <div style="
                        width: 44px;
                        height: 44px;
                        background-color: {c['hex']};
                        border: 1px solid rgba(0,0,0,0.08);
                        border-radius: 8px;
                        flex-shrink: 0;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.10);
                    "></div>
                    <div style="text-align:center; line-height:1.4; width:100%;">{label_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-bottom: 18px;'></div>", unsafe_allow_html=True)


def style_status(val):
    val_str = str(val)
    if val_str == "—":
        return "background-color: #ebebeb; color: #6c757d"
    if val_str == "X":
        return "background-color: #f8d7da; color: #721c24"
    if val_str.endswith("sec "):
        return "background-color: #fff3cd; color: #856404"
    if val_str.endswith("sec"):
        return "background-color: #d4edda; color: #155724"
    return ""


def style_topic(val):
    if str(val):
        return "font-weight: bold; font-size: 15px"
    return ""