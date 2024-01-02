from casadi import *
import numpy as np
import time

line_width = 1.5
fontsize_labels = 12

#updates the state and control sequence based on a given system dynamic
def shift(T, t0, x0, u, f, out, st_prev):
    # Shift the state and control inputs over a time step T
    st = x0
    print("================================", st)
    out_put = out(st)
    print("!!!!!!!!!!", out_put)
    con = u[:, 0].reshape(-1, 1)
    print("cccccccccccccccccccccooooooooooooooooooonnnnnnnnnnnnnnn", con)
    f_value = f(st, con)
    f_value[0] = out_put[0]
    f_value[1] = out_put[1]
    st_ground = st_prev + T * f_value
    print("=+++____________________________", f_value)
    x0 = np.array(st_ground.full())
    t0 = t0 + T
    u0 = np.concatenate((u[1:, :], np.tile(u[-1, :], (1, 1))), axis=0)
    return t0, x0, u0

# Parameters
T = 1  # Sampling interval in seconds
N = 10  # Prediction horizon
a = 10  # Parameter for ellipse area
w_lane = 5.25  # Width of the lane
l_vehicle = 2  # Length of the vehicle
w_vehicle = 2  # Width of the vehicle

# Define symbolic states
Vx = SX.sym('Vx')  # x-axis velocity
Vy = SX.sym('Vy')  # y-axis velocity
x = SX.sym('x')  # x-axis position
y = SX.sym('y')  # y-axis position
theta = SX.sym('theta')  # Orientation angle
vtheta = SX.sym('vtheta')  # Angular velocity
states = vertcat(x, y, Vx, Vy, theta, vtheta)
n_states = states.numel()

# Define symbolic control inputs
ax = SX.sym('ax')  # x-axis acceleration
delta = SX.sym('delta')  # Steering angle
controls = vertcat(ax, delta)
n_controls = controls.numel()
print(n_controls)

# System parameters
caf = 1
car = 1
m = 26
lf = 2
lr = 2
Iz = 10000
g = 1
h = 1
vxtemp = 1

# Define system dynamics coefficients
a1 = -(2 * caf + 2 * car) / (m * vxtemp)
a2 = -vxtemp - (2 * caf * lf - 2 * car * lr) / (m * vxtemp)
a3 = -(2 * caf * lf + 2 * car * lr) / (Iz * vxtemp)
a4 = -(2 * caf * lf * lf + 2 * car * lr * lr) / (Iz * vxtemp)
print("VX!!!!!!!!!!!", a1, a2, a3, a4)

b1 = 2 * caf / m
b2 = 2 * lf * caf / Iz

# Define system matrices
AA = np.array([[0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, a1, 0, a2],
               [0, 0, 0, 0, 0, 1],
               [0, 0, 0, a3, 0, a4]])

BB = np.array([[0, 0],
               [0, 0],
               [1, 0],
               [0, b1],
               [0, 0],
               [0, b2]])

# Define rotation matrix
cos_val = casadi.cos(states[4])
sin_val = casadi.sin(states[4])
CC = casadi.vertcat(casadi.horzcat(cos_val, sin_val),
                    casadi.horzcat(-sin_val, cos_val))
CC_np = casadi.DM(CC).full()
print(states[4])

# Define system dynamics
rhs = AA @ states + BB @ controls
output = CC @ casadi.vertcat(states[2], states[3])
print("%%%%%", output)

f = Function('f', [states, controls], [rhs])
out = Function('out', [states], [output])

# Define decision variables
U = SX.sym('U', n_controls, N)  # Decision variables (control inputs)
P = SX.sym('P', n_states + N * (n_states + n_controls))  # Parameters (initial state, reference states, and reference controls)
X = SX.sym('X', n_states, N + 1)  # State variables (trajectory)

# Cost function
# Weight matrices Q and R
Q = np.zeros((6, 6))
Q[0, 0] = 1  # Weight for x-axis position
Q[1, 1] = 5  # Weight for y-axis position
Q[2, 2] = 100  # Weight for x-axis velocity
Q[3, 3] = 1  # Weight for y-axis velocity
Q[4, 4] = 1  # Weight for orientation angle
Q[5, 5] = 1  # Weight for angular velocity

R = np.zeros((2, 2))
R[0, 0] = 1  # Weight for x-axis acceleration
R[1, 1] = 0.1  # Weight for steering angle

obj = 0  # Cost function
g = []  # Constraints vector
st = X[:, 0]  # Initial state

g = vertcat(g, st - P[0:6])  # Initial condition constraint

# Loop over the prediction horizon
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    P1 = P
    P2 = P[8 * (k + 1) - 2]
    obj = obj + (st - P[8 * (k + 1) - 2:8 * (k + 1) + 4]).T @ Q @ (st - P[8 * (k + 1) - 2:8 * (k + 1) + 4]) + (
                con - P[8 * (k + 1) + 5:8 * (k + 1) + 6]).T @ R @ (con - P[8 * (k + 1) + 5:8 * (k + 1) + 6])  # Cost computation (modified)

    ini_next = X[:, k + 1]
    f_value = f(st, con)
    ini_next_euler = st + (T * f_value)
    g = vertcat(g, ini_next - ini_next_euler)  # Constraint computation

tempg = g
print("?????????????????", tempg)

# Constraints
Vx_min = -1  # m/s   # Change this to a negative value to track the y-axis
Vx_max = 1  # m/s
Vy_min = -2  # m/s
Vy_max = 2  # m/s
x_min = float('-inf')
x_max = float('inf')
y_min = -30
y_max = 300  # ymax limit
theta_min = -3.14
theta_max = 3.14
vtheta_min = -1.57
vtheta_max = 1.57

ax_min = -9
ax_max = 6
delta_min = -1.54
delta_max = 1.57

args = {}
args['lbg'] = [0] * 6 * (N + 1)  # Equality constraints
args['ubg'] = [0] * 6 * (N + 1)  # Equality constraints

# With obstacle
args['lbg'][6 * (N + 1):6 * (N + 1) + (N + 1)] = np.full((N + 1,), -np.inf)  # Inequality constraints
args['ubg'][6 * (N + 1):6 * (N + 1) + (N + 1)] = np.zeros((N + 1,))  # Inequality constraints

# State variable bounds
args['lbx'] = np.zeros((86, 1))  # Lower bounds for state variables (possibly has issues)
args['ubx'] = np.zeros((86, 1))  # Upper bounds for state variables (possibly has issues)

args['lbx'][0:6 * (N + 1):6, 0] = np.tile(x_min, (N + 1,))  # x-axis position lower bound
args['ubx'][0:6 * (N + 1):6, 0] = np.tile(x_max, (N + 1,))  # x-axis position upper bound
args['lbx'][1:6 * (N + 1):6, 0] = np.tile(y_min, (N + 1,))  # y-axis position lower bound
args['ubx'][1:6 * (N + 1):6, 0] = np.tile(y_max, (N + 1,))  # y-axis position upper bound
args['lbx'][2:6 * (N + 1):6, 0] = np.tile(Vx_min, (N + 1,))  # x-axis velocity lower bound
args['ubx'][2:6 * (N + 1):6, 0] = np.tile(Vx_max, (N + 1,))  # x-axis velocity upper bound
args['lbx'][3:6 * (N + 1):6, 0] = np.tile(Vy_min, (N + 1,))  # y-axis velocity lower bound
args['ubx'][3:6 * (N + 1):6, 0] = np.tile(Vy_max, (N + 1,))  # y-axis velocity upper bound
args['lbx'][4:6 * (N + 1):6, 0] = np.tile(theta_min, (N + 1,))  # orientation angle lower bound
args['ubx'][4:6 * (N + 1):6, 0] = np.tile(theta_max, (N + 1,))  # orientation angle upper bound
args['lbx'][5:6 * (N + 1):6, 0] = np.tile(vtheta_min, (N + 1,))  # angular velocity lower bound
args['ubx'][5:6 * (N + 1):6, 0] = np.tile(vtheta_max, (N + 1,))  # angular velocity upper bound

args['lbx'][6 * (N + 1) + 0:6 * (N + 1) + 2 * N:2, 0] = ax_min  # x-axis acceleration lower bound
args['ubx'][6 * (N + 1) + 0:6 * (N + 1) + 2 * N:2, 0] = ax_max  # x-axis acceleration upper bound
args['lbx'][6 * (N + 1) + 1:6 * (N + 1) + 2 * N:2, 0] = delta_min  # steering angle lower bound
args['ubx'][6 * (N + 1) + 1:6 * (N + 1) + 2 * N:2, 0] = delta_max  # steering angle upper bound

args['p'] = np.zeros((86, 1))  # Initialize control parameters (possibly has issues)

# Start MPC
# ------------ Initial values -------------------------
t0 = 0
x0 = np.array([0, 0, Vx_min, 0, 0, 0]).reshape(-1, 1)  # Initial condition
xs = np.array([np.inf, 0, 20, 0]).reshape(-1, 1)  # Reference states
xx = np.zeros((6, 2000))
xx[:, 0:1] = x0
t = np.zeros(10000)
u0 = np.zeros((N, 2))  # Two control inputs for each robot
X0 = np.tile(x0, (N + 1, 1)).reshape(11, 6)  # Initialization of the state decision variables
sim_tim = 200
mpciter = 0
xx1 = np.empty((N + 1, 6, 2000))
u_cl = np.zeros((1, 2))
loop_start = time.time()
Vx_ref = 20
Vx_ref = 20
obs_x = 50
obs_y = 50
diam_safe = 10  # Safe distance
x_prev = np.array([0, 0, 0, 0, 0, 0]).reshape(-1, 1)

while mpciter < sim_tim / T:  # Condition for ending the loop
    # Entrance
    # Add constraints for collision avoidance
    g = tempg
    obs_y = obs_y - 0.2
    print("############", obs_x)
    for k in range(N + 1):
        g = vertcat(g, -np.sqrt(((X[0, k] - obs_x) ** 2) / (5 ** 2) + ((X[1, k] - obs_y) ** 2) / (2 ** 2)) + 1.4)  # Consider ellipse area

    OPT_variables = vertcat(reshape(X, (6 * (N + 1), 1)), reshape(U, (2 * N, 1)))

    nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

    opts = {}
    opts['ipopt.max_iter'] = 2000
    opts['ipopt.print_level'] = 0
    opts['print_time'] = 0
    opts['ipopt.acceptable_tol'] = 1e-8
    opts['ipopt.acceptable_obj_change_tol'] = 1e-6

    solver = nlpsol('solver', 'ipopt', nlp_prob, opts)
    print(mpciter, sim_tim / T)
    current_time = mpciter * T  # Get the current time
    args['p'][0:6] = np.array(x0)[:, 0].reshape(-1, 1)  # Initial condition of the vehicle np.tile(x_min, (N+1,))
    for k in range(1, N + 1):  # Set the reference to track
        t_predict = current_time + (k - 1) * T  # Predicted time instant

        if xx[0, mpciter] + xx[2, mpciter] * (k) * T < 50:  # Lane keeping
            x_ref = 20 * t_predict
            y_ref = 0 * w_lane
            Vx_ref = 2
            Vy_ref = 0
            ax_ref = 0
            delta_ref = 0
            theta_ref = 0
            vtheta_ref = 0
        elif xx[0, mpciter + 1] + xx[2, mpciter + 1] * (k) * T < 500:  # Lane keeping
            x_ref = 50
            y_ref = 4 * t_predict
            Vx_ref = 0
            Vy_ref = 0
            ax_ref = 0
            delta_ref = 0
            theta_ref = 0
            vtheta_ref = 0

        args['p'][8 * (k) - 2:8 * (k) + 4] = np.array([x_ref, y_ref, Vx_ref, Vy_ref, theta_ref, vtheta_ref]).reshape(6, 1)
        args['p'][8 * k + 4:8 * k + 6] = np.array([ax_ref, delta_ref]).reshape(2, 1)

    args['x0'] = np.vstack((X0.T.reshape(6 * (N + 1), 1), u0.T.reshape(2 * N, 1)))
    sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

    u = np.reshape(sol['x'][6 * (N + 1):], (N, 2))
    print("u:ax,delta!!!!!!!!!!!!!!!!??????????????????????????!!!", u)

    xx1[:, :, mpciter + 1] = np.reshape(sol['x'][:6 * (N + 1)], (N + 1, 6))  # Get solution trajectory
    u_cl = np.vstack((u_cl, u[0]))  # Append the first control to the control list

    print("ucl:axchoose,deltachoose.."
          "###########################", u_cl)
    print("XXXXXXXXXXXXXXXXXX"
          "###########################", u_cl[:, 0].tolist())
    print("YYYYYYYYYYYYYYY"
          "###########################", u_cl[:, 1].tolist())
    t[mpciter + 1] = t0

    # Apply the control and shift the solution
    t0, x0, u0 = shift(T, t0, x0, u.T, f, out, x_prev)
    print("ground::::::::::::::::::", x0)
    # Exit, send x0
    x_prev = x0
    vxtemp = x0[2]

    xx[:, mpciter + 1: mpciter + 2] = x0
    X0 = np.reshape(sol['x'][0:6 * (N + 1)], (N + 1, 6))  # Get solution trajectory
    X0 = np.vstack((X0[1:], X0[-1]))  # Shift trajectory to initialize the next step
    mpciter += 1




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Ellipse


class DynamicPlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.x_r_1 = []
        self.y_r_1 = []
        self.line, = self.ax.plot([], [], '-r', linewidth=1.2)
        self.obstacle = Ellipse(xy=(50, 0), width=9, height=3.5, edgecolor='r', fc='red')
        self.obstacle1 = Ellipse(xy=(80, 0), width=9, height=4, edgecolor='r', fc='purple')
        self.vehicle = Rectangle((0, 0), 0, 0, linewidth=1.25, edgecolor='b', facecolor='blue')
        self.ax.add_patch(self.obstacle)
        self.ax.add_patch(self.vehicle)
        self.ax.set_xlim(-1, 200)
        self.ax.set_ylim(-10, 10)

    def __call__(self, k):
        self.ax.clear()
        self.ax.plot([0, 100], [0, 0], '--g', linewidth=1.2)
        self.ax.plot([0, 100], [10, 10], '-b', linewidth=1.2)
        self.ax.plot([0, 100], [-10, -10], '-b', linewidth=1.2)
        self.ax.plot([50, 50], [-50, 50], '--g', linewidth=1.2)
        self.ax.plot([40, 40], [-50, 50], '-b', linewidth=1.2)
        self.ax.plot([60, 60], [-50, 50], '-b', linewidth=1.2)

        x1 = xx[0, k]
        y1 = xx[1, k]
        self.x_r_1.append(x1)
        self.y_r_1.append(y1)
        self.line.set_data(self.x_r_1, self.y_r_1)

        if k < xx.shape[1] - 1:
            self.ax.plot(xx1[0:N, 0, k], xx1[0:N, 1, k], 'r--*')


            last_star_x = xx1[0:N, 0, k][-1]
            last_star_y = xx1[0:N, 1, k][-1]


            self.ax.plot(self.x_r_1, self.y_r_1, '-k', linewidth=1.0)

        self.ax.set_title(f'$N={N},a={a}$', fontsize=fontsize_labels)
        self.ax.set_ylabel('$y$(m)', fontsize=fontsize_labels)
        self.ax.set_xlabel('$x$(m)', fontsize=fontsize_labels)

        self.obstacle.set_center((50, 50 - 0.2*k))  # 更新椭圆的中心位置

        self.ax.add_patch(self.obstacle)
        self.ax.add_patch(self.obstacle1)
        self.ax.add_patch(self.vehicle)
        self.vehicle.set_xy((x1 - l_vehicle / 2, y1 - w_vehicle / 2))
        self.vehicle.set_width(l_vehicle)
        self.vehicle.set_height(w_vehicle)

        return self.line, self.obstacle, self.vehicle

    def start_animation(self):
        ani = FuncAnimation(self.fig, self, frames=range(xx.shape[1]), interval=200)
        plt.show()


# Create an instance object of the DynamicPlot class
dp = DynamicPlot()

# Start the animation of the dynamic graph
dp.start_animation()
