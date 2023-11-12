#Run the command : cd /Users/venugopal.adep/PYTHON/API/AIOPs/Graph; bokeh serve --show graph_jio-5G-tower_cluster.py 
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, HoverTool, Div, PreText
from bokeh.plotting import figure
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np
import random

# Constants
GRID_SIZE = 10
NUM_LOCATIONS = 30  # Number of populated locations

locations = [{"id": f"L{x}_{y}", "x": x * 100, "y": y * 100, "population": random.randint(1, 100)}
             for x, y in random.sample([(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)], NUM_LOCATIONS)]

X = np.array([[loc["x"], loc["y"]] for loc in locations])
populations = np.array([loc["population"] for loc in locations])
kmeans = KMeans(n_clusters=5, random_state=0).fit(X, sample_weight=populations)
centers = kmeans.cluster_centers_

# Calculate pairwise distances and get minimum distance for each cluster center
distances = pairwise_distances(centers)
np.fill_diagonal(distances, np.inf)
min_distances = distances.min(axis=1)

source = ColumnDataSource(data=dict(x=[loc["x"] for loc in locations],
                                    y=[loc["y"] for loc in locations],
                                    population=[loc["population"]/2 for loc in locations]))

tower_center_source = ColumnDataSource(data=dict(x=centers[:, 0], y=centers[:, 1]))
tower_coverage_source = ColumnDataSource(data=dict(x=centers[:, 0], y=centers[:, 1], radius=min_distances / 2))

plot = figure(title="5G Tower Optimization", tools="pan,wheel_zoom,box_zoom,reset", sizing_mode='stretch_both')
plot.circle(x="x", y="y", size="population", source=source, fill_color="blue", line_color="black", alpha=0.6)
plot.circle(x="x", y="y", radius="radius", source=tower_coverage_source, fill_color=None, line_color="red", line_dash="dotted", line_width=2)
plot.circle(x="x", y="y", size=15, source=tower_center_source, fill_color="darkred", line_color="black")

# Add hover tool
hover = HoverTool(tooltips=[("Population", "@population")])
plot.add_tools(hover)

refresh_button = Button(label="Refresh", button_type="default")
tower_count_div = Div(text=f"Number of Towers: {len(centers)}", width=400, height=50, css_classes=["tower-count-style"])

# Custom CSS for styling
css = """
.tower-count-style {
    font-weight: bold;
    font-size: 48px;
    color: blue;
    text-align: center;
    margin: 40px auto;
}
"""
css_widget = PreText(text=css, width=0, height=0)

def refresh():
    global locations, X, populations, centers
    locations = [{"id": f"L{x}_{y}", "x": x * 100, "y": y * 100, "population": random.randint(1, 100)}
                 for x, y in random.sample([(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)], NUM_LOCATIONS)]
    X = np.array([[loc["x"], loc["y"]] for loc in locations])
    populations = np.array([loc["population"] for loc in locations])
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X, sample_weight=populations)
    centers = kmeans.cluster_centers_
    distances = pairwise_distances(centers)
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)
    source.data = dict(x=[loc["x"] for loc in locations],
                       y=[loc["y"] for loc in locations],
                       population=[loc["population"]/2 for loc in locations])
    tower_center_source.data = dict(x=centers[:, 0], y=centers[:, 1])
    tower_coverage_source.data = dict(x=centers[:, 0], y=centers[:, 1], radius=min_distances / 2)

refresh_button.on_click(refresh)

layout = column(css_widget, tower_count_div, plot, row(refresh_button), sizing_mode='stretch_both')
curdoc().add_root(layout)
