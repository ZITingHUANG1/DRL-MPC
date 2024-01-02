import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.animation import FuncAnimation


def myplot(xground, yground):
    fig, ax = plt.subplots()
    ax.plot(xground, yground)

    ax.plot([0, 200], [0, 0], '--g', linewidth=1.2)
    ax.plot([0, 200], [10, 10], '-b', linewidth=1.2)
    ax.plot([0, 200], [-10, -10], '-b', linewidth=1.2)
    ax.plot([100, 100], [-50, 100], '--g', linewidth=1.2)
    ax.plot([115, 115], [-50, 100], '-b', linewidth=1.2)
    ax.plot([85, 85], [-50, 100], '-b', linewidth=1.2)

    # Initial position of the ellipse
    x_ellipse = 100
    y_ellipse = 50
    ellipse = Ellipse(xy=(x_ellipse, y_ellipse), width=10, height=5, edgecolor='r', fc='red')

    # Adding a rectangle at (0, 0) with width and height of 2
    rectangle = Rectangle((-1, -1), 4, 4, edgecolor='b', fc='blue')

    ax.add_patch(ellipse)
    ax.add_patch(rectangle)  # Adding the square to the plot

    def update(frame):
        nonlocal y_ellipse
        y_ellipse -= 0.15  # Move the ellipse 0.15 units along the negative Y axis in each frame
        ellipse.set_center((x_ellipse, y_ellipse))
        return ellipse, rectangle

    ani = FuncAnimation(fig, update, frames=200, interval=200, blit=True)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


x = [0]
y = [0]

myplot(x, y)
