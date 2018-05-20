import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

gf=nx.DiGraph()

gf.add_node("hidden")
gf.add_node("softmax")
gf.add_node("output")
gf.add_node("i2o")
gf.add_node("i2h")
gf.add_node("i2h2")

gf.add_edge("input", "i2o")
gf.add_edge("input", "i2h")
gf.add_edge("hidden", "i2h")
gf.add_edge("hidden", "i2o")
gf.add_edge("i2h", "i2h2")
gf.add_edge("i2h2", "hidden")
gf.add_edge("i2o", "softmax")
gf.add_edge("softmax", "output")


pos={}
pos["input"]=(2,2)
pos["hidden"]=(3,2)
pos["i2h"]=(2.5,3)
pos["i2o"]=(2,3)
pos["i2h2"]=(3,4)
pos["softmax"]=(2,4)
pos["output"]=(2,5)

print(gf.nodes())
# (1, 0.5, 0) orange
# (0.2, 0.5, 0.7) blå
# (0.1, 1.0, 1.0) cyan

# 'input', 'i2h', 'output', 'i2o', 'hidden', 'softmax', 'i2h2']

val_map = {'hidden': (1, 0.5, 0),
          'input': (1, 0.5, 0),
          'output': (1, 0.5, 0),
          'i2o': (0.2, 0.5, 0.7),
          'i2h': (0.2, 0.5, 0.7),
          'i2h2': (0.2, 0.5, 0.7),
          'softmax': (0.1, 1.0, 1.0)}

colors = [val_map.get(node) for node in gf.nodes()]

print(colors)

# [(1, 0.5, 0), (0.2, 0.5, 0.7), (1, 0.5, 0), (0.2, 0.5, 0.7), (1, 0.5, 0), (0.1, 1.0, 1.0), (0.2, 0.5, 0.7)]

nx.draw(gf, with_labels=True, node_color=colors, node_size=1200, pos=pos)
plt.show()


