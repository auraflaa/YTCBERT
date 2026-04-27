import pandas as pd
from visualize_diversity import generate_premium_dashboard

data = [
    {"title": f"Video {i}", "channel_title": "Channel X", "category": "Gaming", "view_count": i * 1000, "duration": i * 60, "url": "http"}
    for i in range(1, 100)
]
df = pd.DataFrame(data)
generate_premium_dashboard(df, 'test_dashboard.html')
print("Successfully generated test_dashboard.html")
