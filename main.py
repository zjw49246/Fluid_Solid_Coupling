# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01

import argparse

import numpy as np

import math

import taichi as ti

# How to run:
#   `python stable_fluid.py`: use the jacobi iteration to solve the linear system.
#   `python stable_fluid.py -S`: use a sparse matrix to do so.
parser = argparse.ArgumentParser()
parser.add_argument('-S',
                    '--use-sp-mat',
                    action='store_true',
                    help='Solve Poisson\'s equation by using a sparse matrix')
parser.add_argument('-a',
                    '--arch',
                    required=False,
                    default="cpu",
                    dest='arch',
                    type=str,
                    help='The arch (backend) to run this example on')
args, unknowns = parser.parse_known_args()

method_jacobi = "jacobi"
method = method_jacobi

test_mode = True
record_video = True
use_MacCormack = True
vc_jacobi_iters = 500
gravity = 0
print_residual = False
grid_in_sq_sample_num_each_row = 4
grid_in_sq_sample_num_each_row_recip = 1 / grid_in_sq_sample_num_each_row
grid_in_sq_sample_dist = 1 / (grid_in_sq_sample_num_each_row - 1)
grid_in_sq_sample_num = grid_in_sq_sample_num_each_row * grid_in_sq_sample_num_each_row
grid_in_sq_sample_num_recip = 1 / grid_in_sq_sample_num
sq_length = 50
# sq_total_sample_num = int((sq_length / grid_in_sq_sample_dist) * (sq_length / grid_in_sq_sample_dist))
sq_total_sample_num = sq_length * sq_length * grid_in_sq_sample_num
sq_total_sample_num_recip = 1 / sq_total_sample_num

res = 512
res_sq_recip = 1.0 / (res * res)
one_third = 1 / 3
one_64 = 1 / 64
dt = 0.03
dx = 0.5
p_jacobi_iters = 500  # 40 for a quicker but less accurate result
f_strength = 10000.0
curl_strength = 0
time_c = 2
maxfps = 60
dye_decay = 1 - 1 / (maxfps * time_c)
force_radius = res / 2.0
debug = False
vc_num_level = 3
top_level_num_grid = (int(res / math.pow(2, vc_num_level))) * (int(res / math.pow(2, vc_num_level)))
rad_45 = ti.math.pi / 4
rho_fluid = 1
rho_square = 7
rho_ratio = rho_square / rho_fluid

use_sparse_matrix = False
arch = "gpu"
if arch in ["x64", "cpu", "arm64"]:
    ti.init(arch=ti.cpu, default_fp=ti.f64)
elif arch in ["cuda", "gpu"]:
    ti.init(arch=ti.cuda, default_fp=ti.f64)
else:
    raise ValueError('Only CPU and CUDA backends are supported for now.')

if method == method_jacobi:
    print(f'Using jacobi iteration {p_jacobi_iters} iterations')

# solid
square_center = ti.Vector.field(2, float, shape=1)
square_rotation = ti.field(float, shape=1)  # (anticlockwise, radian)
square_vel = ti.Vector.field(3, float, shape=1)
square_len = ti.field(float, shape=1)
# square_angular_vel = ti.field(float, shape=1)
square_color = ti.Vector.field(3, float, shape=1)
square_mass = ti.field(float, shape=3)
square_mass_recip = ti.field(float, shape=3)
square_num_sample = ti.field(int, shape=1)

sq_positions = ti.Vector.field(2, float, shape=4)  # (top left, top right, bottom right, bottom left)
grid_in_sq = ti.field(int)
grid_not_in_sq = ti.field(int)
grid_pre_in_sq = ti.field(int)
vel_in_sq = ti.Vector.field(2, int)
grid_in_sq_in_sample_num = ti.field(int)
grid_in_sq_snode = ti.root.dense(ti.ij, (res, res))
grid_in_sq_snode.place(grid_in_sq, grid_not_in_sq, grid_pre_in_sq, vel_in_sq, grid_in_sq_in_sample_num)
vel_u_grid_in_sq_in_sample_num = ti.field(int, shape=(res - 1, res))
vel_v_grid_in_sq_in_sample_num = ti.field(int, shape=(res, res - 1))
J_sq = ti.field(float, shape=(3, res * res))
J_sq_not_zero = ti.field(int, shape=(res * res))
J_sq_not_zero_num = ti.field(int, shape=1)
sparse_mat = ti.Vector.field(3, float, shape=2000000)
sparse_mat_len = ti.field(int, shape=1)
sparse_mat_diag = ti.field(float, shape=(res * res))


_velocities = ti.Vector.field(2, float, shape=(res, res))
_new_velocities = ti.Vector.field(2, float, shape=(res, res))
velocity_divs = ti.field(float, shape=(res, res))
velocity_curls = ti.field(float, shape=(res, res))
_pressures = ti.field(float, shape=(res, res))
_new_pressures = ti.field(float, shape=(res, res))
_dye_buffer = ti.Vector.field(3, float)
_new_dye_buffer = ti.Vector.field(3, float)
_image = ti.Vector.field(3, float)
dye_grid_snode = ti.root.dense(ti.ij, (res, res))
dye_grid_snode.place(_dye_buffer, _new_dye_buffer, _image)

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)

if use_sparse_matrix:
    # use a sparse matrix to solve Poisson's pressure equation.
    @ti.kernel
    def fill_laplacian_matrix(laplacian_matrix: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(res, res):
            row = i * res + j
            center = 0.0
            if j != 0:
                laplacian_matrix[row, row - 1] += -1.0
                center += 1.0
            if j != res - 1:
                laplacian_matrix[row, row + 1] += -1.0
                center += 1.0
            if i != 0:
                laplacian_matrix[row, row - res] += -1.0
                center += 1.0
            if i != res - 1:
                laplacian_matrix[row, row + res] += -1.0
                center += 1.0
            laplacian_matrix[row, row] += center

    N = res * res
    K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
    F_b = ti.ndarray(ti.f32, shape=N)

    fill_laplacian_matrix(K)
    L = K.build()
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(L)
    solver.factorize(L)

@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = ti.max(0, ti.min(res - 1, I))
    return qf[I]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


# 3rd order Runge-Kutta
@ti.func
def backtrace(vf: ti.template(), p, dt_: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt_ * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt_ * v2
    v3 = bilerp(vf, p2)
    p -= dt_ * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        if use_MacCormack:
            p_back = backtrace(vf, p, dt)
            p_back_and_forth = backtrace(vf, p_back, -dt)
            p_error = (p_back_and_forth - p)
            p = p_back + p_error
        else:
            p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p) * dye_decay


@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template(),
                  imp_data: ti.types.ndarray()):
    g_dir = -ti.Vector([0, gravity]) * 300
    for i, j in vf:
        omx, omy = imp_data[2], imp_data[3]
        mdir = ti.Vector([imp_data[0], imp_data[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        # dv = F * dt
        factor = ti.exp(-d2 / force_radius)

        dc = dyef[i, j]
        a = dc.norm()

        momentum = (mdir * f_strength * factor + g_dir * a / (1 + a)) * dt
        # momentum = (mdir * f_strength * factor) * dt

        v = vf[i, j]
        vf[i, j] = v + momentum
        # add dye
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (4 / (res / 15)**2)) * ti.Vector(
                [imp_data[4], imp_data[5], imp_data[6]])

        dyef[i, j] = dc


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == res - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == res - 1:
            vt.y = -vc.y
        velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5


@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        velocity_curls[i, j] = (vr.y - vl.y - vt.x + vb.x) * 0.5


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= dt * ti.Vector([pr - pl, pt - pb]) / rho_fluid


@ti.kernel
def enhance_vorticity(vf: ti.template(), cf: ti.template()):
    # anti-physics visual enhancement...
    for i, j in vf:
        cl = sample(cf, i - 1, j)
        cr = sample(cf, i + 1, j)
        cb = sample(cf, i, j - 1)
        ct = sample(cf, i, j + 1)
        cc = sample(cf, i, j)
        force = ti.Vector([abs(ct) - abs(cb),
                           abs(cl) - abs(cr)]).normalized(1e-3)
        force *= curl_strength * cc
        vf[i, j] = ti.min(ti.max(vf[i, j] + force * dt, -1e3), 1e3)


@ti.kernel
def copy_divergence(div_in: ti.template(), div_out: ti.types.ndarray()):
    for I in ti.grouped(div_in):
        div_out[I[0] * res + I[1]] = -div_in[I]


@ti.kernel
def apply_pressure(p_in: ti.types.ndarray(), p_out: ti.template()):
    for I in ti.grouped(p_out):
        p_out[I] = p_in[I[0] * res + I[1]]

@ti.kernel
def compute_square_positions():
    center_x, center_y, length, theta = square_center[0][0], square_center[0][1], square_len[0], square_rotation[0]
    lc = length * ti.math.sqrt(2) * 0.5
    center = ti.Vector([center_x, center_y])
    sq_positions[0][0] = center_x - lc * ti.math.sin(theta + rad_45)
    sq_positions[0][1] = center_y + lc * ti.math.cos(theta + rad_45)
    sq_positions[1][0] = center_x + lc * ti.math.sin(-theta + rad_45)
    sq_positions[1][1] = center_y + lc * ti.math.cos(-theta + rad_45)
    sq_positions[2] = 2 * center - sq_positions[0]
    sq_positions[3] = 2 * center - sq_positions[1]

@ti.func
def check_point_in_square(point_x: float, point_y: float):
    result = True
    for i in ti.ndrange(4):
        j = (i + 1) % 4
        sq_x1, sq_y1, sq_x2, sq_y2 = sq_positions[i][0], sq_positions[i][1], sq_positions[j][0], sq_positions[j][1]
        x1, y1 = sq_x1 - point_x, sq_y1 - point_y
        x2, y2 = sq_x2 - point_x, sq_y2 - point_y
        cross = x1 * y2 - x2 * y1
        if cross > 0:
            result = False
            break
    return result

@ti.kernel
def check_grid_in_square():
    square_num_sample[0] = 0
    for i, j in grid_in_sq:
        if check_point_in_square(i + 0.5, j + 0.5):
            grid_in_sq[i, j] = 1
            grid_not_in_sq[i, j] = 0
        else:
            grid_in_sq[i, j] = 0
            grid_not_in_sq[i, j] = 1
        grid_in_sq_in_sample_num[i, j] = 0
        for offset_i, offset_j in ti.ndrange(grid_in_sq_sample_num_each_row, grid_in_sq_sample_num_each_row):
            x, y = i + offset_i * grid_in_sq_sample_dist, j + offset_j * grid_in_sq_sample_dist
            if check_point_in_square(x, y):
                grid_in_sq_in_sample_num[i, j] += 1
        square_num_sample[0] += grid_in_sq_in_sample_num[i, j]
    # print(f'{square_num_sample[0]} {sq_total_sample_num}')
    real_square_len = square_len[0] * dx
    square_mass[0] = real_square_len * real_square_len * rho_square * square_num_sample[0] * sq_total_sample_num_recip
    square_mass[1] = square_mass[0]
    square_mass[2] = square_mass[0] * real_square_len * real_square_len / 6
    square_mass_recip[0] = 1 / (square_mass[0] + 1e-5)
    square_mass_recip[1] = 1 / (square_mass[1] + 1e-5)
    square_mass_recip[2] = 1 / (square_mass[2] + 1e-5)

@ti.kernel
def check_velocity_in_square():
    for i, j in vel_in_sq:
        if check_point_in_square(i, j + 0.5):
            vel_in_sq[i, j][0] = 1
        else:
            vel_in_sq[i, j][0] = 0

        if check_point_in_square(i + 0.5, j):
            vel_in_sq[i, j][1] = 1
        else:
            vel_in_sq[i, j][1] = 0
@ti.kernel
def check_velocity_grid_in_square():
    for i, j in vel_u_grid_in_sq_in_sample_num:
        vel_u_grid_in_sq_in_sample_num[i, j] = 0
        for offset_i, offset_j in ti.ndrange(grid_in_sq_sample_num_each_row, grid_in_sq_sample_num_each_row):
            x, y = i + 0.5 + offset_i * grid_in_sq_sample_dist, j + offset_j * grid_in_sq_sample_dist
            if check_point_in_square(x, y):
                vel_u_grid_in_sq_in_sample_num[i, j] += 1

    for i, j in vel_v_grid_in_sq_in_sample_num:
        vel_v_grid_in_sq_in_sample_num[i, j] = 0
        for offset_i, offset_j in ti.ndrange(grid_in_sq_sample_num_each_row, grid_in_sq_sample_num_each_row):
            x, y = i + offset_i * grid_in_sq_sample_dist, j + 0.5 + offset_j * grid_in_sq_sample_dist
            if check_point_in_square(x, y):
                vel_v_grid_in_sq_in_sample_num[i, j] += 1

@ti.kernel
def advect_square():
    square_center[0][0] += square_vel[0][0] * dt
    square_center[0][1] += square_vel[0][1] * dt
    square_rotation[0] += square_vel[0][2] * dt

@ti.kernel
def apply_gravity_to_square():
    grav = ti.Vector([0.0, -9.8])
    square_vel[0][0] += grav[0] * dt
    square_vel[0][1] += grav[1] * dt

@ti.kernel
def fluid_square_collision():
    for i, j in grid_in_sq:
        if grid_in_sq[i, j] == 1:
            new_i = int(i + square_vel[0][0] * dt)
            new_j = int(j + square_vel[0][1] * dt)
            dyes_pair.cur[new_i, new_j] = dyes_pair.cur[i, j] #- ti.Vector([0, 0.5, 0.5])

            # pos = ti.Vector([i, j])
            # old_pos = int(backtrace(velocities_pair.cur, pos, dt * 0.75))
            # dyes_pair.cur[old_pos[0], old_pos[1]] = dyes_pair.cur[i, j] - ti.Vector([0, 0.5, 0.5])
            old_i = int(i - square_vel[0][0] * dt)
            old_j = int(j - square_vel[0][1] * dt)
            dyes_pair.cur[old_i, old_j] = dyes_pair.cur[i, j] #- ti.Vector([0, 0.5, 0.5])

            dyes_pair.cur[i, j] = ti.Vector([0, 0, 0])
            pressures_pair.cur[i, j] = 0.0

        # for grid which is not in square but was in square at prev step, use neighboring grids' dye to fill it
        if grid_in_sq[i, j] == 0 and grid_pre_in_sq[i, j] == 1:
            # dyes_pair.cur[i, j] = ti.Vector([1, 1, 1])
            for offset_i, offset_j in ti.ndrange((-1, 2), (-1, 2)):
                if offset_i == 0 and offset_j == 0:
                    continue
                nb_i, nb_j = i + offset_i, j + offset_j
                if nb_i < 0 or nb_i >= res or nb_j < 0 or nb_j >= res:
                    continue
                if grid_in_sq[nb_i, nb_j] == 1 or grid_pre_in_sq[nb_i, nb_j] == 1:
                    continue
                dyes_pair.cur[i, j] = ti.math.max(dyes_pair.cur[i, j], dyes_pair.cur[nb_i, nb_j]) #- ti.Vector([0, 0.5, 0.5])

        grid_pre_in_sq[i, j] = grid_in_sq[i, j]

    for i, j in vel_in_sq:
        if vel_in_sq[i, j][0] == 1:
            _velocities[i, j][0] = square_vel[0][0]
        if vel_in_sq[i, j][1] == 1:
            _velocities[i, j][1] = square_vel[0][1]

    # restrict square in the window
    for i in sq_positions:
        border = 2
        x, y = sq_positions[i][0], sq_positions[i][1]
        if x <= border:
            square_vel[0][0] = -square_vel[0][0]
        if x >= res - border:
            square_vel[0][0] = -square_vel[0][0]
        if y <= border:
            square_vel[0][1] = -square_vel[0][1]
        if y >= res - border:
            square_vel[0][1] = -square_vel[0][1]

@ti.kernel
def generate_J_sq():
    J_sq.fill(0.0)
    for i, j in vel_u_grid_in_sq_in_sample_num:
        if 0 < vel_u_grid_in_sq_in_sample_num[i, j] < grid_in_sq_sample_num:
            row = i * res + j
            mul = vel_u_grid_in_sq_in_sample_num[i, j] * grid_in_sq_sample_num_recip * dx
            J_sq[0, row] += mul * grid_not_in_sq[i, j]
            J_sq[0, row + res] += -mul * grid_not_in_sq[i + 1, j]
    for i, j in vel_v_grid_in_sq_in_sample_num:
        if 0 < vel_v_grid_in_sq_in_sample_num[i, j] < grid_in_sq_sample_num:
            row = i * res + j
            mul = vel_v_grid_in_sq_in_sample_num[i, j] * grid_in_sq_sample_num_recip * dx
            J_sq[1, row] += mul * grid_not_in_sq[i, j]
            J_sq[1, row + 1] += -mul * grid_not_in_sq[i, j + 1]
    for i, j in grid_in_sq_in_sample_num:
        if 0 < grid_in_sq_in_sample_num[i, j] < grid_in_sq_sample_num:
            x, y = i + 0.5, j + 0.5
            x_relative, y_relative = x - square_center[0][0], y - square_center[0][1]
            row = i * res + j
            mul = grid_in_sq_in_sample_num[i, j] * grid_in_sq_sample_num_recip * dx * dx
            J_sq[2, row] += mul * grid_not_in_sq[i, j] * (-y_relative + x_relative)
            if i < res - 1:
                J_sq[2, row + res] += mul * grid_not_in_sq[i + 1, j] * y_relative
            if j < res - 1:
                J_sq[2, row + 1] += -mul * grid_not_in_sq[i, j + 1] * x_relative

@ti.kernel
def generate_sparse_mat():
    # initialize sparse_mat and sparse_mat_diag
    sparse_mat_len[0] = 0
    for i in sparse_mat_diag:
        sparse_mat_diag[i] = 0.0

    J_sq_not_zero_num[0] = 0
    for i in ti.ndrange(res * res):
        for k in range(3):
            if J_sq[k, i] > 1e-5 or J_sq[k, i] < -1e-5:
                idx = ti.atomic_add(J_sq_not_zero_num[0], 1)
                J_sq_not_zero[idx] = i
                break

    # square KE
    for m_id, n_id in ti.ndrange(J_sq_not_zero_num[0], J_sq_not_zero_num[0]):  # m = (i, j)
        m, n = J_sq_not_zero[m_id], J_sq_not_zero[n_id]
        tmp = 0.0

        for k in range(3):
            tmp += dt * J_sq[k, m] * square_mass_recip[k] * J_sq[k, n]
        # print(tmp)
        if tmp > 1e-5 or tmp < -1e-5:
            if n == m:
                sparse_mat_diag[n] += tmp
            else:
                idx = ti.atomic_add(sparse_mat_len[0], 1)
                sparse_mat[idx][0], sparse_mat[idx][1], sparse_mat[idx][2] = m, n, tmp

    # fluid KE
    for m, n_id in ti.ndrange(res * res, 5):
        n = m
        if n_id == 1:
            n += res
        elif n_id == 2:
            n -= res
        elif n_id == 3:
            n += 1
        elif n_id == 4:
            n -= 1

        tmp = 0.0

        if n == m:
            i, j = m // res, m % res
            if i < res - 1:
                tmp += 1 - vel_u_grid_in_sq_in_sample_num[i, j] * grid_in_sq_sample_num_recip
            if i > 0:
                tmp += 1 - vel_u_grid_in_sq_in_sample_num[i - 1, j] * grid_in_sq_sample_num_recip
            if j < res - 1:
                tmp += 1 - vel_v_grid_in_sq_in_sample_num[i, j] * grid_in_sq_sample_num_recip
            if j > 0:
                tmp += 1 - vel_v_grid_in_sq_in_sample_num[i, j - 1] * grid_in_sq_sample_num_recip
        elif n == m + res:
            i, j = m // res, m % res
            if i < res - 1:
                tmp += -1 + vel_u_grid_in_sq_in_sample_num[i, j] * grid_in_sq_sample_num_recip
        elif n == m - res:
            i, j = m // res, m % res
            if i > 0:
                tmp += -1 + vel_u_grid_in_sq_in_sample_num[i - 1, j] * grid_in_sq_sample_num_recip
        elif n == m + 1:
            i, j = m // res, m % res
            if j < res - 1:
                tmp += -1 + vel_v_grid_in_sq_in_sample_num[i, j] * grid_in_sq_sample_num_recip
        elif n == m - 1:
            i, j = m // res, m % res
            if j > 0:
                tmp += -1 + vel_v_grid_in_sq_in_sample_num[i, j - 1] * grid_in_sq_sample_num_recip

        tmp *= dt / rho_fluid
        # tmp *= dt * 0.25
        if tmp > 1e-5 or tmp < -1e-5:
            if n == m:
                sparse_mat_diag[n] += tmp
            else:
                idx = ti.atomic_add(sparse_mat_len[0], 1)
                sparse_mat[idx][0], sparse_mat[idx][1], sparse_mat[idx][2] = m, n, tmp

    # print(sparse_mat_len[0])

@ti.kernel
def jacobi_iteration_sparse_mat(pf: ti.template(), new_pf: ti.template()):
    for i, j in new_pf:
        new_pf[i, j] = 0.0
        # fluid
        if i < res - 1:
            new_pf[i, j] += (-1 + vel_u_grid_in_sq_in_sample_num[i, j] * grid_in_sq_sample_num_recip) * velocities_pair.cur[i + 1, j][0]
        if i > 0:
            new_pf[i, j] += (1 - vel_u_grid_in_sq_in_sample_num[i - 1, j] * grid_in_sq_sample_num_recip) * velocities_pair.cur[i, j][0]
        if j < res - 1:
            new_pf[i, j] += (-1 + vel_v_grid_in_sq_in_sample_num[i, j] * grid_in_sq_sample_num_recip) * velocities_pair.cur[i, j + 1][1]
        if j > 0:
            new_pf[i, j] += (1 - vel_v_grid_in_sq_in_sample_num[i, j - 1] * grid_in_sq_sample_num_recip) * velocities_pair.cur[i, j][1]

        new_pf[i, j] *= dx

        # square
        row = i * res + j
        for k in range(3):
            new_pf[i, j] += -J_sq[k, row] * square_vel[0][k]

    for i in ti.ndrange(sparse_mat_len[0]):
        m, n, val = int(sparse_mat[i][0]), int(sparse_mat[i][1]), sparse_mat[i][2]
        # print(m, n, val)
        p_i, p_j = n // res, n % res
        new_p_i, new_p_j = m // res, m % res
        new_pf[new_p_i, new_p_j] -= val * pf[p_i, p_j]

    for i in sparse_mat_diag:
        new_p_i, new_p_j = i // res, i % res
        new_pf[new_p_i, new_p_j] *= 1 / (sparse_mat_diag[i] + 1e-5)
    # print(new_pf[10, 10])

def solve_jacobi_sparse_mat():
    for _ in range(p_jacobi_iters):
        jacobi_iteration_sparse_mat(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

@ti.kernel
def apply_pressure_on_square():
    for dim, p_i in J_sq:
        i, j = p_i // res, p_i % res
        square_vel[0][dim] += dt * square_mass_recip[dim] * J_sq[dim, p_i] * pressures_pair.cur[i, j]


def step(mouse_data):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    velocities_pair.swap()
    dyes_pair.swap()
    advect_square()

    compute_square_positions()
    check_grid_in_square()
    check_velocity_grid_in_square()
    check_velocity_in_square()
    fluid_square_collision()


    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)
    apply_gravity_to_square()
    generate_J_sq()
    apply_pressure_on_square()

    divergence(velocities_pair.cur)

    if curl_strength:
        vorticity(velocities_pair.cur)
        enhance_vorticity(velocities_pair.cur, velocity_curls)

    generate_sparse_mat()
    solve_jacobi_sparse_mat()
    # fluid_square_collision()

    # if method == method_mgpcg:
    #     solve_pressure_MGPCG()
    # elif method == method_cg:
    #     solve_pressure_conjugate_gradients()
    # elif method == method_jacobi:
    #     solve_pressure_jacobi()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)

    if debug:
        divergence(velocities_pair.cur)
        div_s = np.sum(velocity_divs.to_numpy())
        print(f'divergence={div_s}')


class MouseDataGen:
    def __init__(self, test_mode=False):
        self.prev_mouse = None
        self.prev_color = None
        self.test_mode = test_mode
        self.test_frame = 0
        self.max_test_frame = 1000

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if self.test_mode:
            self.test_frame += 1
            if self.test_frame % 1 == 0:
                # mxy = np.array([0.1 * self.test_frame, 0.15 * self.test_frame], dtype=np.float32) * res
                mxy = np.array([0.5, 0], dtype=np.float32) * res
                if self.prev_mouse is None:
                    self.prev_mouse = mxy
                    # Set lower bound to 0.3 to prevent too dark colors
                    self.prev_color = (np.array([1, 1, 1]) * 0.7) + 0.3
                else:
                    mdir = ti.Vector([0.1 * ti.math.sin(self.test_frame * 5 / 180 * ti.math.pi), 1])
                    mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                    mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                    mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                    mouse_data[4:7] = self.prev_color
                    self.prev_mouse = mxy
            return mouse_data

        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None

        return mouse_data


def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    dyes_pair.cur.fill(0)
    initialize_square()

def initialize_square():
    square_center[0][0] = res * 0.5
    square_center[0][1] = res * 0.3
    square_vel[0][0] = 0.0
    square_vel[0][1] = 0.0
    square_vel[0][2] = 0.0
    square_len[0] = sq_length
    square_rotation[0] = 0
    square_color[0] = ti.Vector([0, 1, 1])
    real_square_len = square_len[0] * dx
    square_mass[0] = real_square_len * real_square_len * rho_square
    square_mass[1] = square_mass[0]
    square_mass[2] = square_mass[0] * real_square_len * real_square_len / 6
    square_mass_recip[0] = 1 / (square_mass[0] + 1e-5)
    square_mass_recip[1] = 1 / (square_mass[1] + 1e-5)
    square_mass_recip[2] = 1 / (square_mass[2] + 1e-5)

@ti.kernel
def generate_image():
    for i, j in _image:
        if grid_in_sq[i, j] == 0:
            _image[i, j] = dyes_pair.cur[i, j]
        else:
            _image[i, j] = square_color[0]

def main():
    global debug, curl_strength, method, test_mode, num_iters_label
    visualize_d = True  #visualize dye (default)
    visualize_v = False  #visualize velocity
    visualize_c = False  #visualize curl
    visualize_p = False  # visualize curl

    paused = False

    initialize_square()

    gui = ti.GUI('Stable Fluid', (res, res))
    md_gen = MouseDataGen(test_mode)

    # method_jacobi_button = gui.button('1.Jacobi')
    # method_cg_button = gui.button('2.CG')
    # method_mgpcg_button = gui.button('3.MGPCG')
    # test_mode_button = gui.button('example On/Off')
    # method_label = gui.label('iteration method')
    # num_iters_label = gui.label('iteration num')

    result_dir = f'./results_rho{rho_square}_sq_size{sq_length}'
    video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    while gui.running:
        # if method == method_jacobi:
        #     method_label.value = 1
        #     num_iters_label.value = p_jacobi_iters
        # elif method == method_cg:
        #     method_label.value = 2
        #     num_iters_label.value = p_conjugate_gradients_iters
        # elif method == method_mgpcg:
        #     method_label.value = 3
        #     num_iters_label.value = p_MGPCG_iters

        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == 'r':
                paused = False
                reset()
            elif e.key == 's':
                if curl_strength:
                    curl_strength = 0
                else:
                    curl_strength = 7
            elif e.key == 'v':
                visualize_v = True
                visualize_c = False
                visualize_d = False
                visualize_p = False
            elif e.key == 'd':
                visualize_d = True
                visualize_v = False
                visualize_c = False
                visualize_p = False
            elif e.key == 'c':
                visualize_c = True
                visualize_d = False
                visualize_v = False
                visualize_p = False
            elif e.key == 'p':
                visualize_c = False
                visualize_d = False
                visualize_v = False
                visualize_p = True
            elif e.key == gui.SPACE:
                paused = not paused
            elif e.key == 'd':
                debug = not debug


        # Debug divergence:
        # print(max((abs(velocity_divs.to_numpy().reshape(-1)))))

        if not paused:
            mouse_data = md_gen(gui)
            # print(mouse_data)
            step(mouse_data)
        if visualize_c:
            vorticity(velocities_pair.cur)
            gui.set_image(velocity_curls.to_numpy() * 0.03 + 0.5)
        elif visualize_d:
            generate_image()
            gui.set_image(_image)
            if record_video:
                video_manager.write_frame(_image)
        elif visualize_v:
            gui.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
        elif visualize_p:
            gui.set_image(pressures_pair.cur.to_numpy() * 0.03 + 0.5)
        gui.show()

    if record_video:
        print()
        print('Exporting .mp4 and .gif videos...')
        video_manager.make_video(gif=True, mp4=True)
        print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
        print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')


if __name__ == '__main__':
    main()