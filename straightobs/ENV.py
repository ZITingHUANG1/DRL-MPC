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
    ellipse = Ellipse(xy=(50, 0), width=20, height=10, edgecolor='r', fc='red')
    ellipse2 = Ellipse(xy=(100, 0.5), width=4, height=4, edgecolor='r', fc='yellow')  # 添加新椭圆
    rectangle = Rectangle((-1, -1), 2, 2, edgecolor='b', fc='blue')  # Adding a blue square at (0, 0)

    ax.add_patch(ellipse)
    ax.add_patch(ellipse2)  # Add the second ellipse to the plot
    ax.add_patch(rectangle)  # Adding the square to the plot

    def update(frame):
        # Calculate new center coordinates for the ellipse
        x = 50 + frame * 0  # Move the ellipse 0.2 units along the X axis in each frame
        ellipse.set_center((x, 0))
        return ellipse, ellipse2, rectangle

    ani = FuncAnimation(fig, update, frames=len(xground), interval=200, blit=True)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

x = [0]
y = [0]
myplot(x, y)

