from matplotlib.pyplot import *
import scipy.special as sp

def plt_3d(x, y, z):
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, label='title',cmap='viridis', edgecolor='none')
    show()

def integral(n, step_f, rs_f, ys_f, rs_F):
    # first dimension - x
    r_2d = np.broadcast_to(rs_f[:, np.newaxis], (n, n))

    # second dimension - u
    u_2d = np.broadcast_to(rs_F[np.newaxis, :], (n, n))

    # J0(kr) * r
    A = sp.j0(u_2d * r_2d) * r_2d

    # scale rows by f(x)
    A = A * np.broadcast_to(ys_f[:, np.newaxis], (n, n))

    int_weights = np.ones(n)
    int_weights[0] = 1 / 2
    int_weights[-1] = 1 / 2
    int_weights *= step_f

    # scale rows by int_weights
    A = A * np.broadcast_to(int_weights[:, np.newaxis], (n, n))

    ys_F = np.sum(A, axis=0)

    return ys_F

def draw_2d(sp_n, sp_m, sp_c, xs, ys, s):
    extent = [xs[0], xs[-1], xs[0], xs[-1]]
    subplot(sp_n, sp_m, sp_c)
    imshow(np.abs(ys), extent=extent)
    colorbar()
    title(f'$\\left|{s}\\right|$')

def get_2d(F, shape, dtype):
    F2d = np.zeros(shape, dtype=dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            F2d[i][j] = F[j + i * shape[1]]
    return F2d

n = 50
alpha = 6.0
beta = 6.0
x = np.linspace(-np.pi, np.pi,   n)
y = np.linspace(-np.pi, np.pi, n)

x2d, y2d = np.meshgrid(x, y)
f = lambda r: np.exp((-r ** 2) / beta) * (np.sin(alpha * r) ** 2)
r2d = np.sqrt(np.sqrt(x2d ** 2 + y2d ** 2))
f2d = f(r2d)

figure(figsize=(8, 6))
#plt_3d(x2d, y2d, np.abs(f2d))
draw_2d(2, 2, 1, x, f2d, 'sourse f')

r = r2d.ravel()
F = integral(r.shape[0], abs(r[1] - r[0]), r, f2d.ravel(), r)
F2d = get_2d(F, x2d.shape, np.complex128)

#plt_3d(x2d, y2d, np.abs(F2d))
draw_2d(2, 2, 2, x, F2d, 'my Hankel')

show()