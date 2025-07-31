import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import numpy as np

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')
x, y = [], []

for i in range(50):
    x.append(i)
    y.append(np.sin(i / 5))
    line.set_data(x, y)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)

plt.ioff()
plt.show()
