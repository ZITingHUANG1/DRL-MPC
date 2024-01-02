import numpy as np


def vehicle_dynamic(y, y_dot, fai, fai_dot,vx,delta):
    caf = 1
    car = 1
    m = 10
    lf = 2
    lr = 2
    Iz = 10
    g = 1
    h = 1

    a = -(2 * caf + 2 * car) / (m * vx)
    b = -vx - (2 * caf * lf - 2 * car * lr) / (m * vx)
    c = -(2 * caf * lf - 2 * car * lr) / (Iz * vx)
    d = -(2 * caf * lf * lf + 2 * car * lr * lr) / (Iz * vx)

    A = np.array([[0, 1, 0, 0],
                  [0, a, 0, b],
                  [0, 0, 0, 1],
                  [0, c, 0, d]])
    e = 2 * caf / m
    f = 2 * lf * caf / Iz
    B = np.array([[0],
                  [e],
                  [0],
                  [f]])

    state = np.array([[y], [y_dot], [fai], [fai_dot]])
    C = np.array([[0], [g], [0], [h]])

    new_state = np.dot(A, state) + np.dot(B, delta)

    return new_state.flatten()
