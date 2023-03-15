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
method_cg = "cg"
method_mgpcg = "mgpcg"
method = method_jacobi

test_mode = True
use_MacCormack = True
# use_conjugate_gradients = False
use_warm_starting = True
use_jacobi_preconditioner = True
# use_MGPCG = True
p_conjugate_gradients_iters = 20
p_MGPCG_iters = 5
vc_jacobi_iters = 500
gravity = 0
print_residual = False
grid_in_sq_sample_num_each_row = 4
grid_in_sq_sample_num_each_row_recip = 1 / grid_in_sq_sample_num_each_row
grid_in_sq_sample_dist = 1 / (grid_in_sq_sample_num_each_row - 1)
grid_in_sq_sample_num = grid_in_sq_sample_num_each_row * grid_in_sq_sample_num_each_row
grid_in_sq_sample_num_recip = 1 / grid_in_sq_sample_num
sq_length = 30
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
rho_square = 9
rho_ratio = rho_square / rho_fluid

use_sparse_matrix = False
arch = "gpu"
if arch in ["x64", "cpu", "arm64"]:
    ti.init(arch=ti.cpu, default_fp=ti.f64)
elif arch in ["cuda", "gpu"]:
    ti.init(arch=ti.cuda, default_fp=ti.f64)
else:
    raise ValueError('Only CPU and CUDA backends are supported for now.')

if method == method_cg:
    print(f'Using conjugate gradients {p_conjugate_gradients_iters} iterations')
elif method == method_mgpcg:
    print('Using MGPCG')
elif method == method_jacobi:
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
vel_in_sq = ti.Vector.field(2, int)
grid_in_sq_in_sample_num = ti.field(int)
grid_in_sq_snode = ti.root.dense(ti.ij, (res, res))
grid_in_sq_snode.place(grid_in_sq, vel_in_sq, grid_in_sq_in_sample_num)
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
# A_cg = ti.field(int, shape=(res * res, 5))
r_cg_cur = ti.field(float, shape=(res * res))
r_cg_nxt = ti.field(float, shape=(res * res))
p_cg_cur = ti.field(float, shape=(res * res))
p_cg_nxt = ti.field(float, shape=(res * res))
Ap_cg = ti.field(float, shape=(res * res))
x_cg = ti.field(ti.f64, shape=(res * res))
u_vc = ti.field(float)
b_vc = ti.field(float)
vc_snode = ti.root.dense(ti.i, vc_num_level + 1).dense(ti.j, res * res) # total of (vc_num_level + 1) levels
vc_snode.place(u_vc)
vc_snode.place(b_vc)
u_vc_top_level = ti.field(float, shape=top_level_num_grid)
r_vc = ti.field(float, shape=(res * res))
rho_MGPCG = ti.field(ti.f64, shape=1)
alpha_MGPCG = ti.field(ti.f64, shape=1)

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)
r_cg_pair = TexPair(r_cg_cur, r_cg_nxt)
p_cg_pair = TexPair(p_cg_cur, p_cg_nxt)

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

# if use_conjugate_gradients:
#     @ti.kernel
#     def fill_laplacian_matrix(laplacian_matrix: ti.template()):
#         for i, j in ti.ndrange(res, res):
#             row = i * res + j
#             count = 0
#             if j != 0:
#                 count += 1
#                 laplacian_matrix[row, count] = row - 1
#             if j != res - 1:
#                 count += 1
#                 laplacian_matrix[row, count] = row + 1
#             if i != 0:
#                 count += 1
#                 laplacian_matrix[row, count] = row - res
#             if i != res - 1:
#                 count += 1
#                 laplacian_matrix[row, count] = row + res
#             laplacian_matrix[row, 0] = row
#             if count < 4:
#                 for k in range(count + 1, 5):
#                     laplacian_matrix[row, k] = -1
#     fill_laplacian_matrix(A_cg)
#     print(A_cg)

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

@ti.func
def laplacian_A_mul_p(p: ti.template(), Ap: ti.template(), res: int, mul_ct: float, mul_nb: float):
    # ct: center, nb: neighbor
    for row in Ap:
        i = row // res
        j = row % res
        Ap[row] = 0.0
        mul = 1.0
        if method == method_cg and use_jacobi_preconditioner:
            mul = 0.25
        Ap[row] += mul * mul_ct * p[row]
        if j != 0:
            Ap[row] += mul * mul_nb * p[row - 1]
        if j != res - 1:
            Ap[row] += mul * mul_nb * p[row + 1]
        if i != 0:
            Ap[row] += mul * mul_nb * p[row - res]
        if i != res - 1:
            Ap[row] += mul * mul_nb * p[row + res]

@ti.func
def vc_laplacian_A_mul_p(p: ti.template(), Ap: ti.template(), res: int, mul_ct: float, mul_nb: float, l: int):
    # ct: center, nb: neighbor
    for i, j in ti.ndrange(res, res):
        row = i * res + j
        Ap[row] = 0.0
        # if row >= res * res:
        #     continue

        # i = row // res
        # j = row % res
        mul = 1.0
        if method == method_cg and use_jacobi_preconditioner:
            mul = 0.25
        Ap[row] += mul * mul_ct * p[l, row]
        if j != 0:
            Ap[row] += mul * mul_nb * p[l, row - 1]
        if j != res - 1:
            Ap[row] += mul * mul_nb * p[l, row + 1]
        if i != 0:
            Ap[row] += mul * mul_nb * p[l, row - res]
        if i != res - 1:
            Ap[row] += mul * mul_nb * p[l, row + res]

@ti.kernel
def pressure_conjugate_gradients_pre(pf: ti.template()):
    for i, j in pf:
        row = i * res + j
        if use_warm_starting:
            pressure = sample(pf, i, j)
            x_cg[row] = pressure
        else:
            x_cg[row] = 0

    laplacian_A_mul_p(x_cg, Ap_cg, res, 4.0, -1.0)

    for i, j in pf:
        row = i * res + j
        r_cg_pair.cur[row] = -velocity_divs[i, j]
        if use_jacobi_preconditioner:
            r_cg_pair.cur[row] *= 0.25
        r_cg_pair.cur[row] -= Ap_cg[row]
        # # pressure = sample(pf, A_cg[row, 0][0], A_cg[row, 0][1])
        # # r_cg[row] -= 4 * pressure
        # for k in range(5):
        #     mul = -1.0
        #     if k == 0:
        #         mul = 4.0
        #     if use_jacobi_preconditioner:
        #         mul *= 0.25
        #     if A_cg[row, k] != -1:
        #         r_cg_pair.cur[row] -= mul * x_cg[A_cg[row, k]]
        #     else:
        #         break

    for r in p_cg_pair.cur:
        p_cg_pair.cur[r] = r_cg_pair.cur[r]

@ti.kernel
def pressure_conjugate_gradients_iter(pf: ti.template()):
    laplacian_A_mul_p(p_cg_pair.cur, Ap_cg, res, 4.0, -1.0)
    # for r in Ap_cg:
    #     Ap_cg[r] = 0
    #     for k in range(5):
    #         mul = -1.0
    #         if k == 0:
    #             mul = 4.0
    #         if use_jacobi_preconditioner:
    #             mul *= 0.25
    #         # print(f'{k} {mul}')
    #         if A_cg[r, k] != -1:
    #             Ap_cg[r] += mul * p_cg_pair.cur[A_cg[r, k]]

    nume = 0.0
    deno = 0.0
    for r in r_cg_pair.cur:
        nume += r_cg_pair.cur[r] * r_cg_pair.cur[r]
        deno += p_cg_pair.cur[r] * Ap_cg[r]
    alpha = nume / (deno + 1e-5)

    r_cg_pair_cur_norm_sq = 0.0
    r_cg_pair_nxt_norm_sq = 0.0
    for r in x_cg:
        x_cg[r] += alpha * p_cg_pair.cur[r]
        r_cg_pair.nxt[r] = r_cg_pair.cur[r] - alpha * Ap_cg[r]
        r_cg_pair_cur_norm_sq += r_cg_pair.cur[r] * r_cg_pair.cur[r]
        r_cg_pair_nxt_norm_sq += r_cg_pair.nxt[r] * r_cg_pair.nxt[r]

    if print_residual:
        print(ti.sqrt(r_cg_pair_nxt_norm_sq))
    # if ti.sqrt(r_cg_pair_nxt_norm_sq) < 1e-5:
    #     return

    beta = r_cg_pair_nxt_norm_sq / (r_cg_pair_cur_norm_sq + 1e-5)
    for r in p_cg_pair.nxt:
        p_cg_pair.nxt[r] = r_cg_pair.nxt[r] + beta * p_cg_pair.cur[r]
        p_cg_pair.cur[r] = p_cg_pair.nxt[r]
        r_cg_pair.cur[r] = r_cg_pair.nxt[r]

@ti.kernel
def pressure_conjugate_gradients_post(new_pf: ti.template()):
    for i, j in new_pf:
        row = i * res + j
        new_pf[i, j] = x_cg[row]

@ti.func
def calculate_vc_res(l: int):
    grid_length = ti.pow(2, l)
    vc_res = int(res / grid_length)

    return vc_res

@ti.func
def damped_jacobi_smooth(l: int):
    vc_res = calculate_vc_res(l)
    vc_laplacian_A_mul_p(u_vc, Ap_cg, vc_res, 4.0, -1.0, l)
    for i in ti.ndrange(vc_res * vc_res):
        # if i >= vc_res * vc_res:
        #     continue
        u_vc[l, i] += 2 * one_third * (b_vc[l, i] - Ap_cg[i]) * 0.25

@ti.kernel
def v_cycle_pre():
    for row in ti.ndrange(res * res):
        u_vc[0, row] = 0.0

@ti.func
def v_cycle_restrict(l: int):
    vc_res_cur = calculate_vc_res(l)
    vc_res_nxt = calculate_vc_res(l + 1)
    for i, j in ti.ndrange(vc_res_nxt, vc_res_nxt):
        row = i * vc_res_nxt + j
        b_vc[l + 1, row] = 0.0
        # get the left bottom u_h, origin at the most left bottom point
        h_i = i * 2
        h_j = j * 2
        count = 0
        mul = 3
        for m, n in ti.static(ti.ndrange((-1, 3), (-1, 3))):
            i_cur = i + m
            j_cur = j + n
            row_cur = i_cur * vc_res_cur + j_cur
            if 0 <= i_cur < vc_res_cur and 0 <= j_cur < vc_res_cur:
                if 0 <= m <= 1 and 0 <= n <= 1:
                    mul = 9
                elif m == n or (m == -1 and n == 2) or (m == 2 and n == -1):
                    mul = 1
                b_vc[l + 1, row] += mul * r_vc[row_cur]
                count += mul
        b_vc[l + 1, row] /= (count + 1e-5)

@ti.kernel
def v_cycle_up_iter(l: int):
    vc_res = calculate_vc_res(l)
    # print(vc_res)
    damped_jacobi_smooth(l)
    vc_laplacian_A_mul_p(u_vc, Ap_cg, vc_res, 4.0, -1.0, l)
    for i in ti.ndrange(vc_res * vc_res):
        r_vc[i] = 0
        # if i >= vc_res * vc_res:
        #     continue
        r_vc[i] = b_vc[l, i] - Ap_cg[i]
    v_cycle_restrict(l)

    vc_res_nxt = calculate_vc_res(l + 1)
    for row in ti.ndrange(vc_res_nxt * vc_res_nxt):
        u_vc[l + 1, row] = 0.0


@ti.kernel
def v_cycle_solve():
    vc_res = calculate_vc_res(vc_num_level)
    for row in ti.ndrange(vc_res * vc_res):
        u_vc_top_level[row] = u_vc[vc_num_level, row]
    for i, j in ti.ndrange(vc_res, vc_res):
        row = i * vc_res + j
        ul = 0.0
        ur = 0.0
        ub = 0.0
        ut = 0.0
        if j != 0:
            ul = u_vc_top_level[row - 1]
        if j != vc_res - 1:
            ur = u_vc_top_level[row + 1]
        if i != 0:
            ub = u_vc_top_level[row - res]
        if i != vc_res - 1:
            ut = u_vc_top_level[row + res]
        b_val = b_vc[vc_num_level, row]
        u_vc[vc_num_level, row] = (ul + ur + ub + ut - b_val) * 0.25

@ti.func
def v_cycle_prolongate(l: int):
    vc_res_prev = calculate_vc_res(l + 1)
    vc_res_cur = calculate_vc_res(l)
    for i, j in ti.ndrange(vc_res_cur, vc_res_cur):
        row = i * vc_res_cur + j
        h2_i = i // 2
        h2_j = j // 2
        # decide which four u_2h to find
        h2_range_mul_y = 1
        h2_range_mul_x = 1
        if i == h2_i * 2:
            h2_range_mul_y = -1
        if j == h2_j * 2:
            h2_range_mul_x = -1
        mul = 3
        count = 0
        sum = 0.0
        for m, n in ti.static(ti.ndrange((0, 2), (0, 2))):
            i_prev = h2_i + h2_range_mul_y * m
            j_prev = h2_j + h2_range_mul_x * n
            row_prev = i_prev * vc_res_prev + j_prev
            if 0 <= i_prev < vc_res_prev and 0 <= j_prev < vc_res_prev:
                if m == 0 and n == 0:
                    mul = 9
                elif m == 1 and n == 1:
                    mul = 1
                sum += mul * u_vc[l + 1, row_prev]
                count += mul
        u_vc[l, row] += sum / (count + 1e-5)


@ti.kernel
def v_cycle_down_iter(l: int):
    v_cycle_prolongate(l)
    damped_jacobi_smooth(l)

@ti.kernel
def v_cycle_smooth():
    damped_jacobi_smooth(0)

def v_cycle():
    v_cycle_pre()
    for l in range(vc_num_level):
        v_cycle_up_iter(l)
    for _ in range(vc_jacobi_iters):
        v_cycle_solve()
    for l in range(vc_num_level - 1, -1, -1):
        v_cycle_down_iter(l)
    for _ in range(10):
        v_cycle_smooth()

@ti.kernel
def pressure_MGPCG_pre_v_cycle(pf: ti.template()):
    for i, j in pf:
        row = i * res + j
        if use_warm_starting:
            pressure = sample(pf, i, j)
            x_cg[row] = pressure
        else:
            x_cg[row] = 0

    laplacian_A_mul_p(x_cg, Ap_cg, res, 4.0, -1.0)
    # print(Ap_cg[0])
    mu = 0.0
    for i, j in pf:
        row = i * res + j
        r_cg_pair.cur[row] = -velocity_divs[i, j]
        if method == method_cg and use_jacobi_preconditioner:
            r_cg_pair.cur[row] *= 0.25
        r_cg_pair.cur[row] -= Ap_cg[row]
        mu += r_cg_pair.cur[row] * res_sq_recip

    # mu *= res_sq_recip
    # print(mu)
    for row in r_cg_pair.cur:
        r_cg_pair.cur[row] -= mu
        b_vc[0, row] = r_cg_pair.cur[row]

@ti.kernel
def pressure_MGPCG_post_v_cycle():
    rho_MGPCG[0] = 0.0
    for row in p_cg_pair.cur:
        p_cg_pair.cur[row] = u_vc[0, row]
        rho_MGPCG[0] += p_cg_pair.cur[row] * r_cg_pair.cur[row]
        # r_cg_pair.cur[row] = b_vc[0, row]

@ti.kernel
def pressure_MGPCG_iter_pre_v_cycle():
    laplacian_A_mul_p(p_cg_pair.cur, Ap_cg, res, 4.0, -1.0)
    sigma = 0.0
    for row in Ap_cg:
        sigma += p_cg_pair.cur[row] * Ap_cg[row]
    alpha_MGPCG[0] = rho_MGPCG[0] / (sigma + 1e-5)
    mu = 0.0
    r_norm_sq = 0.0
    for row in r_cg_pair.cur:
        r_cg_pair.cur[row] -= alpha_MGPCG[0] * Ap_cg[row]
        mu += r_cg_pair.cur[row] * res_sq_recip
        r_norm_sq += r_cg_pair.cur[row] * r_cg_pair.cur[row]

    if print_residual:
        print(ti.sqrt(r_norm_sq))
    # mu *= res_sq_recip
    for row in r_cg_pair.cur:
        r_cg_pair.cur[row] -= mu
        b_vc[0, row] = r_cg_pair.cur[row]

@ti.kernel
def pressure_MGPCG_iter_post_v_cycle():
    rho_new = 0.0
    for row in r_cg_pair.cur:
        # r_cg_pair.cur[row] = b_vc[0, row]
        rho_new += u_vc[0, row] * r_cg_pair.cur[row]
    beta = rho_new / (rho_MGPCG[0] + 1e-5)
    rho_MGPCG[0] = rho_new
    for row in x_cg:
        x_cg[row] += alpha_MGPCG[0] * p_cg_pair.cur[row]
        p_cg_pair.cur[row] = u_vc[0, row] + beta * p_cg_pair.cur[row]

@ti.kernel
def pressure_MGPCG_post(new_pf: ti.template()):
    for i, j in new_pf:
        row = i * res + j
        new_pf[i, j] = x_cg[row]

def solve_pressure_sp_mat():
    copy_divergence(velocity_divs, F_b)
    x = solver.solve(F_b)
    apply_pressure(x, pressures_pair.cur)


def solve_pressure_jacobi():
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

def solve_pressure_conjugate_gradients():
    pressure_conjugate_gradients_pre(pressures_pair.cur)
    for _ in range(p_conjugate_gradients_iters):
        pressure_conjugate_gradients_iter(pressures_pair.cur)
    pressure_conjugate_gradients_post(pressures_pair.nxt)
    pressures_pair.swap()

def solve_pressure_MGPCG():
    pressure_MGPCG_pre_v_cycle(pressures_pair.cur)
    v_cycle()
    pressure_MGPCG_post_v_cycle()

    for _ in range(p_MGPCG_iters):
        pressure_MGPCG_iter_pre_v_cycle()
        v_cycle()
        pressure_MGPCG_iter_post_v_cycle()

    pressure_MGPCG_post(pressures_pair.nxt)
    pressures_pair.swap()

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
        else:
            grid_in_sq[i, j] = 0
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
            dyes_pair.cur[new_i, new_j] = dyes_pair.cur[i, j]
            dyes_pair.cur[i, j] = ti.Vector([0, 0, 0])
            pressures_pair.cur[i, j] = 0.0

    for i, j in vel_in_sq:
        if vel_in_sq[i, j][0] == 1:
            _velocities[i, j][0] = square_vel[0][0]
        if vel_in_sq[i, j][1] == 1:
            _velocities[i, j][1] = square_vel[0][1]

    for i in sq_positions:
        border = 2
        x, y = sq_positions[i][0], sq_positions[i][1]
        if x <= border:
            square_vel[0][0] = 100.0
        if x >= res - border:
            square_vel[0][0] = -100.0
        if y <= border:
            square_vel[0][1] = 100.0
        if y >= res - border:
            square_vel[0][1] = -100.0

@ti.kernel
def generate_J_sq():
    J_sq.fill(0.0)
    for i, j in vel_u_grid_in_sq_in_sample_num:
        if 0 < vel_u_grid_in_sq_in_sample_num[i, j] < grid_in_sq_sample_num:
            row = i * res + j
            mul = vel_u_grid_in_sq_in_sample_num[i, j] * grid_in_sq_sample_num_recip * dx
            J_sq[0, row] += mul
            J_sq[0, row + res] += -mul
    for i, j in vel_v_grid_in_sq_in_sample_num:
        if 0 < vel_v_grid_in_sq_in_sample_num[i, j] < grid_in_sq_sample_num:
            row = i * res + j
            mul = vel_v_grid_in_sq_in_sample_num[i, j] * grid_in_sq_sample_num_recip * dx
            J_sq[1, row] += mul
            J_sq[1, row + 1] += -mul
    for i, j in grid_in_sq_in_sample_num:
        if 0 < grid_in_sq_in_sample_num[i, j] < grid_in_sq_sample_num:
            x, y = i + 0.5, j + 0.5
            x_relative, y_relative = x - square_center[0][0], y - square_center[0][1]
            row = i * res + j
            mul = grid_in_sq_in_sample_num[i, j] * grid_in_sq_sample_num_recip * dx * dx
            J_sq[2, row] += mul * (-y_relative + x_relative)
            if i < res - 1:
                J_sq[2, row + res] += mul * y_relative
            if j < res - 1:
                J_sq[2, row + 1] += -mul * x_relative

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

    divergence(velocities_pair.cur)

    if curl_strength:
        vorticity(velocities_pair.cur)
        enhance_vorticity(velocities_pair.cur, velocity_curls)

    generate_J_sq()
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

def change_iters_num(iter_method, increase=True):
    global p_jacobi_iters, p_conjugate_gradients_iters, p_MGPCG_iters
    mul = 1
    if not increase:
        mul = -1
    if iter_method == method_jacobi:
        p_jacobi_iters += 10 * mul
        if p_jacobi_iters < 10:
            p_jacobi_iters = 10
        elif p_jacobi_iters > 1000:
            p_jacobi_iters = 1000
    elif iter_method == method_cg:
        p_conjugate_gradients_iters += 1 * mul
        if p_conjugate_gradients_iters < 1:
            p_conjugate_gradients_iters = 1
        elif p_conjugate_gradients_iters > 300:
            p_conjugate_gradients_iters = 300
    elif iter_method == method_mgpcg:
        p_MGPCG_iters += 1 * mul
        if p_MGPCG_iters < 1:
            p_MGPCG_iters = 1
        elif p_MGPCG_iters > 50:
            p_MGPCG_iters = 50

def initialize_square():
    square_center[0][0] = res * 0.5
    square_center[0][1] = res * 0.9
    square_vel[0][0] = 0.0
    square_vel[0][1] = 0.0
    square_vel[0][2] = 0.1
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
            # elif e.key == method_jacobi_button:
            #     # reset()
            #     method = method_jacobi
            #     num_iters_label.value = p_jacobi_iters
            # elif e.key == method_cg_button:
            #     # reset()
            #     method = method_cg
            #     num_iters_label.value = p_conjugate_gradients_iters
            # elif e.key == method_mgpcg_button:
            #     # reset()
            #     method = method_mgpcg
            #     num_iters_label.value = p_MGPCG_iters
            # elif e.key == test_mode_button:
            #     reset()
            #     test_mode = not test_mode
            #     md_gen = MouseDataGen(test_mode)
            # elif e.key == gui.UP:
            #     change_iters_num(method, True)
            # elif e.key == gui.DOWN:
            #     change_iters_num(method, False)


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
        elif visualize_v:
            gui.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
        elif visualize_p:
            gui.set_image(pressures_pair.cur.to_numpy() * 0.03 + 0.5)
        gui.show()


if __name__ == '__main__':
    main()