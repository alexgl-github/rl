import matplotlib.pyplot as plt
import numpy as np

class Plot:

    def __init__(self, num_subplots=1, title=None, subplot_title=None):
        plt.ion()
        self.fig = plt.figure()
        plt.title(title)

        self.ax = []
        for idx in range(num_subplots):
            ax = plt.subplot(num_subplots, 1, idx+1)
            self.ax.append(ax)

        self.x = []
        self.y = []
        for idx in range(num_subplots):
            self.y.append(np.zeros((0,)))
            self.x.append(np.zeros((0,)))

        self.line = []
        for idx in range(num_subplots):
            line, = self.ax[idx].plot(self.x[idx], self.y[idx], 'b-')
            self.line.append(line)

        for idx in range(num_subplots):
             self.ax[idx].title.set_text(subplot_title[idx])

    def update(self, value, idx):
        self.y[idx] = np.append(self.y[idx], value)
        self.x[idx] = [i for i in range(0, len(self.y[idx]))]
        self.ax[idx].relim()
        self.ax[idx].autoscale_view()
        self.line[idx].set_ydata(self.y[idx])
        self.line[idx].set_xdata(self.x[idx])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


