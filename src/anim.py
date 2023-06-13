import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# number of pedestrians
n = 1000
# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(8, 4))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 8), ax.set_xticks([])
ax.set_ylim(0, 4), ax.set_yticks([])

# read results from C++
with open('soa/example.txt') as f:
    pos = [[float(x) for x in line.split()] for line in f]
pos = np.array(pos).reshape(-1, n, 3)
# pos: (iterations, num_pedestrians, posX posY color)

color = np.ones((n, 4))
color[:, 0] = pos[0, :, 2]
color[:, 1] = 1-pos[0, :, 2]
color[:, 2] = pos[0, :, 2]
scat = ax.scatter(pos[0, :, 0], pos[0, :, 1],
                  s=30, lw=0.5, edgecolors=color,
                  facecolors='none')


def update(frame_number):
    current_index = frame_number % pos.shape[0]
    scat.set_offsets(pos[current_index, :, :2])


# Construct the animation, using the update function as the animation director.
animation = FuncAnimation(fig, update, interval=0.001, save_count=100)
plt.show()
