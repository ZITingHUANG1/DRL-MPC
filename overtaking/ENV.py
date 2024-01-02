import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.animation import FuncAnimation

def myplot(xground, yground):
    fig, ax = plt.subplots()
    ax.plot(xground, yground)
    ax.set_ylim(-35, 35)  # Adjust the y-axis limits
    ax.plot([0, 200], [0, 0], '--g', linewidth=1.2)
    ax.plot([0, 200], [10, 10], '-b', linewidth=1.2)
    ax.plot([0, 200], [-10, -10], '-b', linewidth=1.2)
    ellipse = Ellipse(xy=(30, 0), width=12, height=6, edgecolor='r', fc='red')

    rectangle = Rectangle((-1, -1), 2, 2, edgecolor='b', fc='blue')  # Adding a blue square at (0, 0)
    ax.add_patch(ellipse)
    ax.add_patch(rectangle)  # Adding the square to the plot

    def update(frame):
        # Calculate new center coordinates for the ellipse
        x = 30 + frame * 0.2  # Move the ellipse 0.2 units along the X axis in each frame
        ellipse.set_center((x, 0))
        return ellipse, rectangle

    ani = FuncAnimation(fig, update, frames=len(xground), interval=200, blit=True)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

xg = [0, 0.01]
yg = [0, 0.01]

myplot(xg, yg)
