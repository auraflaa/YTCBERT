import pandas as pd
import plotly.graph_objects as go
import json

df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'C'],
    'channel_title': ['Ch1', 'Ch2', 'Ch1', 'Ch3', 'Ch2'],
    'title': ['T1', 'T2', 'T3', 'T4', 'T5'],
    'view_count_num': [1000, 2000, 0, 4000, 5000],
    'duration_sec': [100, 200, 300, 0, 500]
})

fig_tier = go.Figure(go.Histogram(x=df['view_count_num'], nbinsx=50))
tier_html = fig_tier.to_html(full_html=False, include_plotlyjs=False)

fig_scatter = go.Figure()
valid_df = df[(df['duration_sec'] > 0) & (df['view_count_num'] > 0)]
for cat in valid_df['category'].unique():
    cat_df = valid_df[valid_df['category'] == cat]
    fig_scatter.add_trace(go.Scatter(
        x=cat_df['duration_sec'], 
        y=cat_df['view_count_num'], 
        mode='markers', 
        name=cat
    ))
scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs=False)

html = f"<html><body>{tier_html}{scatter_html}</body></html>"
with open('test_plotly_output.html', 'w') as f:
    f.write(html)
print("Dumped.")
