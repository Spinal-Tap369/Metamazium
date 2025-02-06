# metamazium\env\dynamics.py

import numpy
from numba import njit

PI_4 = 0.7853981
PI_2 = 1.5707963
PI = 3.1415926
t_PI = 6.2831852
PI2d = 57.29578

OFFSET_10 = numpy.asarray([0.5, 0.5], dtype="float32")
OFFSET_01 = numpy.asarray([-0.5, 0.5], dtype="float32")
OFFSET_m0 = numpy.asarray([-0.5, -0.5], dtype="float32")
OFFSET_0m = numpy.asarray([0.5, -0.5], dtype="float32")


@njit(cache=True)
def nearest_point(pos, line_1, line_2):
    unit_ori = line_2 - line_1
    edge_norm = numpy.sqrt(numpy.sum(unit_ori * unit_ori))
    unit_ori /= max(1.0e-6, edge_norm)

    dist_1 = numpy.sum((pos - line_1) * unit_ori)
    if(dist_1 > edge_norm):
        return numpy.sqrt(numpy.sum((pos - line_2) ** 2)), numpy.copy(line_2)
    elif(dist_1 < 0):
        return numpy.sqrt(numpy.sum((pos - line_1) ** 2)), numpy.copy(line_1)
    else:
        line_p = line_1 + dist_1 * unit_ori
        return numpy.sqrt(numpy.sum((pos - line_p) ** 2)), numpy.copy(line_p)

@njit(cache=True)
def vector_move(ori, turn_rate, walk_speed, dt):
    ori += turn_rate * dt
    dx = walk_speed * numpy.cos(ori) * dt
    dy = walk_speed * numpy.sin(ori) * dt
    return ori, numpy.array([dx, dy], dtype="float32")

@njit(cache=True)
def collision_force(dist_vec, cell_size, col_dist):
    dist = float(numpy.sqrt(numpy.sum(dist_vec * dist_vec)))
    eff_col_dist = col_dist / cell_size
    if(dist > 0.708 + eff_col_dist):
        return numpy.array([0.0, 0.0], dtype="float32")
    if(abs(dist_vec[0]) < 0.5 and abs(dist_vec[1]) < 0.5):
        return numpy.float32(0.50 / max(dist, 1.0e-6) * (0.708 + eff_col_dist - dist) * cell_size) * dist_vec
    # Additional logic for corner collisions can be added here
    return numpy.float32(0.50 * (eff_col_dist - dist) * cell_size) * dist_vec / dist

@njit(cache=True)
def vector_move_with_collision(ori, pos, turn_rate, walk_speed, dt, cell_walls, cell_size, col_dist):
    slide_factor = 0.20
    collision_occurred = False  # Initialize collision flag
    if(walk_speed < 0):
        walk_speed *= 0.50
    tmp_pos = numpy.copy(numpy.array(pos, dtype="float32"))
    for _ in range(int(100 * dt)):
        ori, offset = vector_move(ori, turn_rate, walk_speed, 0.01)
        exp_pos = tmp_pos + offset
        exp_cell = exp_pos / cell_size
        
        # Consider collisions in the new cell
        col_f = numpy.array([0.0, 0.0], dtype="float32")
        for i in range(-1, 2): 
            for j in range(-1, 2): 
                w_i = i + int(exp_cell[0])
                w_j = j + int(exp_cell[1])
                if(w_i > -1 and w_i < cell_walls.shape[0] and w_j > -1  and w_j < cell_walls.shape[1]):
                    if(cell_walls[w_i, w_j] > 0):
                        cell_deta = exp_cell - numpy.floor(exp_cell) - numpy.array([i + 0.5, j + 0.5], dtype="float32")
                        force = collision_force(cell_deta, cell_size, col_dist)
                        if(numpy.any(force != 0)):
                            collision_occurred = True  # Collision detected
                        col_f += force
        tmp_pos = col_f + exp_pos
    return ori, tmp_pos, collision_occurred  # Return collision flag