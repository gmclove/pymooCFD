from pymooCFD.core.cfdCase import YALES2Case
from pymooCFD.util.handleData import saveTxt

import h5py
import os
import random
import pygmsh
# import gmsh
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.repair import Repair


import numpy as np
# from pymooCFD.core.pymooBase import CFDGeneticProblem
#
# class Place_AP_Problem(CFDGeneticProblem):
#     def __init__(self, CFDCase, n_APs=2, **kwargs):
#         super().__init__(CFDCase, **kwargs)
#         self.n_APs = n_APs
#         self.
#         self.gaps =
#         self.ALPHABET = [c for c in string.ascii_lowercase]
#
#     def _evaluate(self, x, out, *args, **kwargs):
#         n_a, n_b = 0, 0
#         for c in x[0]:
#             if c == 'a':
#                 n_a += 1
#             elif c == 'b':
#                 n_b += 1
#
#         out["F"] = np.array([- n_a, - n_b], dtype=float)
#
# from pymoo.core.sampling import Sampling
#
# class MySampling(Sampling):
#
#     def _do(self, problem, n_samples, **kwargs):
#         X = np.full((n_samples, 1), None, dtype=object)
#
#         for i in range(n_samples):
#             X[i, 0] = "".join([np.random.choice(problem.ALPHABET) for _ in range(problem.n_characters)])
#
#         return X
#


class RemoveBadPlacement(Repair):
    @staticmethod
    def moveAP(x_coor, y_coor):
        # Channel parameters
        dom_x0, dom_y0 = 0, 0
        dom_dx, dom_dy = 4, 4
        centers = [[1.25, 1, 0], [2.75, 1, 0],
                   [1.25, 2, 0], [2.75, 2, 0],
                   [1.25, 3, 0], [2.75, 3, 0]]
        r = 0.125
        ap_l = [0.3, 0.3]
        # spacing = ap_l[0]
        wall_space_dx = ap_l[0] # + spacing
        wall_space_dy = ap_l[1] # + spacing
        min_x, min_y = dom_x0 + wall_space_dx, dom_y0 + wall_space_dy
        max_x, max_y = dom_dx + dom_x0 - wall_space_dx, dom_y0 + dom_dy - wall_space_dy
        person_space_x, person_space_y = r + ap_l[0], r + ap_l[1]
        if x_coor < min_x:
            x_coor = min_x
        if x_coor > max_x:
            x_coor = max_x
        if y_coor < min_y:
            y_coor = min_y
        if y_coor > max_y:
            y_coor = max_y
        for cir_cent in centers:
            cir_cx = cir_cent[0]
            cir_cy = cir_cent[1]
            x_max, x_min = cir_cx + person_space_x, cir_cx - person_space_x
            y_max, y_min = cir_cy + person_space_y, cir_cy - person_space_y
            x_new = [x_min, x_max]
            y_new = [y_min, y_max]
            if x_coor < x_max and x_coor > x_min:
                x_coor = random.choice(x_new)
            if y_coor < y_max and y_coor > y_min:
                y_coor = random.choice(y_new)
        return x_coor, y_coor

    def _do(self, problem, pop, **kwargs):
        X = pop.get('X')
        for ind, x in enumerate(X):
            x[0], x[1] = self.moveAP(x[0], x[1])
            X[ind] = x
        pop.set('X', X)
        return pop


class Room2D_AP(YALES2Case):
    base_case_path = os.path.join(os.path.dirname(__file__), 'base_cases',
                                  '2D_room-ap-base')
    n_var = 4
    var_labels = ['x-Location [m]', 'y-Location [m]',
                  'Direction', 'AP ACH [m^2/hr]']
    var_type = ['real', 'real', 'int', 'real']
    xl = [0.5, 0.5, 1, 0.5]
    xu = [3.5, 3.5, 4, 6]

    n_obj = 2
    obj_labels = ['Room ACH [m^2/hr]', 'Subjects Mean Exposure']

    n_constr = 0

    repair = RemoveBadPlacement()

    externalSolver = True
    procLim = 70
    nProc = 10
    # solverExecCmd = ['mpirun', '-n', str(nProc), '2D_room']
    solverExecCmd = ['sbatch', '--wait', 'jobslurm.sh']
    onlyParallelizeSolve = True

    def __init__(self, case_path, x, meshSF=1, **kwargs):
        super().__init__(case_path, x, meshSF=meshSF,
                         inputFile='2D_room.in',
                         meshFile='room_6P_ap.msh22',
                         datFile='temporal.h5',
                         jobFile='jobslurm.sh',
                         **kwargs)

    def _preProc(self):
        self.genMesh()
        direct = self.x[2]
        ap_ach = self.x[3]  # 0.0296
        ap_l = 0.3
        room_area = 16  # m^2
        ap_speed = ap_ach * room_area / 3600 / ap_l
        if direct == 1:
            u_str = f'0. {ap_speed} 0.'
            walls = 'x'
            inlets = 'y'
        elif direct == 2:
            u_str = f'{ap_speed} 0. 0.'
            walls = 'y'
            inlets = 'x'
        elif direct == 3:
            u_str = f'0. -{ap_speed} 0.'
            walls = 'x'
            inlets = 'y'
        elif direct == 4:
            u_str = f'-{ap_speed} 0. 0.'
            walls = 'y'
            inlets = 'x'
        else:
            raise Exception(f'Can\'t handle direction parameter - {direct}')
        in_lines = self.input_lines_rw
        bnd_wall_ids = [f'unit1{walls}{surf}' for surf in range(2)]
        bnd_inlet_ids = [f'unit1{inlets}{surf}' for surf in range(2)]
        for id in bnd_wall_ids:
            kw_lines = self.findKeywordLines(id, in_lines)
            for line_i, _ in kw_lines:
                del in_lines[line_i]
            bnd_lines = [f"BOUNDARY {id} DESCRIPTION = 'ap-{id}'",
                         f'BOUNDARY {id} TYPE = WALL']
            in_lines.extend(bnd_lines)

        for id in bnd_inlet_ids:
            kw_lines = self.findKeywordLines(id, in_lines)
            for line_i, _ in kw_lines:
                del in_lines[line_i]
            bnd_lines = [f"BOUNDARY {id} DESCRIPTION = 'ap-{id}'",
                         f'BOUNDARY {id} TYPE = INLET',
                         f'BOUNDARY {id} U = ' + u_str,
                         f'BOUNDARY {id} Z = 0.0',
                         f'BOUNDARY {id} PHI_NORMAL_FLUX = 0.0'
                         ]
            in_lines.extend(bnd_lines)
        self.input_lines_rw = in_lines


        self.job_lines_rw = [
            '#!/bin/bash',
            "#SBATCH --partition=ib --constraint='ib&haswell_1'",
            '#SBATCH --cpus-per-task=1',
            '#SBATCH --ntasks=20',
            '#SBATCH --time=30:00:00',
            '#SBATCH --mem-per-cpu=2G',
            '#SBATCH --job-name=ap_room',
            '#SBATCH --output=slurm.out',
            'module use $HOME/yales2/modules && module load $(cd $HOME/yales2/modules; ls)',
            'cd $SLURM_SUBMIT_DIR',
            'mpirun 2D_room'
            ]

        # for line_i, line in lines:
        #     in_lines[line_i] = ''
        #     if 'DESCRIPTION' in line:
        #         in_lines[line_i] = f"BOUNDARY {id} DESCRIPTION = 'ap-{id}'"
        #     elif 'TYPE' in line:
        #         in_lines[line_i] = f'BOUNDARY {id} TYPE = INLET'
        #     elif 'U' in line:
        #         in_lines[line_i] = f'BOUNDARY {id} U = '+u_str
        #     elif 'Z' in line:
        #         in_lines[line_i] = f'BOUNDARY {id} Z = 0.0'
        #     elif 'PHI' in line:
        #         in_lines[line_i] = f'BOUNDARY {id} PHI_NORMAL_FLUX = 0.0'
        #     else:
        #         in_lines[line_i] = ''

    def _postProc(self):
        with h5py.File(self.datPath, 'r') as f:
            t = f['Datas']['TOTAL_TIME'][:]
            SP1 = f['Datas']['Surf_Int1'][:]
            SP2 = f['Datas']['Surf_Int2'][:]
            SP3 = f['Datas']['Surf_Int3'][:]
            SP4 = f['Datas']['Surf_Int4'][:]
            SP5 = f['Datas']['Surf_Int5'][:]
            SP6 = f['Datas']['Surf_Int6'][:]
        # tot_dt = t[-1] - t[0]
        SPs = [SP1, SP2, SP3, SP4, SP5, SP6]
        t_start = 1800
        mask = np.where(t > t_start)
        t = t[mask]
        # dt = t[-1] - t_start
        SPs = [sp[mask] for sp in SPs]
        t_wghts = [t[i] - t[i - 1] for i in range(1, len(t))]
        t_mid = [np.mean([t[i], t[i - 1]]) for i in range(1, len(t))]
        emit_person = 3
        t_avgs = []
        for j, sp in enumerate(SPs):
            p = j + 1
            if not p == emit_person:
                plt.plot(t, sp)
                plt.title(f'Passive Scalar Surface Average - Person {p}')
                plt.xlabel('Time [sec]')
                plt.ylabel('Passive Scalar Surface Average [unitless]')
                path = os.path.join(self.abs_path, f'sp-p{p}.png')
                plt.savefig(path)
                plt.clf()
                sp_mid = [np.mean(np.abs([sp[i], sp[i - 1]]))
                          for i in range(1, len(sp))]
                plt.plot(t_mid, sp_mid)
                plt.title(
                    f'Passive Scalar Surface Average - Person {p}: mid-points')
                plt.xlabel('Time [sec]')
                plt.ylabel('Passive Scalar Surface Average [unitless]')
                path = os.path.join(self.abs_path, f'sp_mid-p{p}.png')
                plt.savefig(path)
                plt.clf()
                # plt.show()
                t_avgs.append(np.average(sp_mid, weights=t_wghts))
        saveTxt(self.abs_path, 'surf-avg-scalar.txt', t_avgs)
        tot_avg = np.mean(t_avgs)
        ACH = 2 + self.x[3]
        self.f = [ACH, tot_avg]
        return self.f

    def _genMesh(self):
        resolution = 0.01
        coarsening = 2
        # Channel parameters
        # dom_x0, dom_y0, dom_z0 = 0, 0, 0
        dom_dx = 4
        dom_dy = 4
        W_vent = 0.5
        p_in = 0.125
        # c = [1, 1, 0]
        centers = [[1.25, 1, 0], [2.75, 1, 0],
                   [1.25, 2, 0], [2.75, 2, 0],
                   [1.25, 3, 0], [2.75, 3, 0]]
        r = 0.125
        # inlet_c = [-0.1, dom_dy/2, 0]

        ap_l = [0.3, 0.3]
        ap_center = [self.x[0], self.x[1]]
        # Initialize empty geometry using the build in kernel in GMSH
        geometry = pygmsh.geo.Geometry()
        # Fetch model we would like to add data to
        model = geometry.__enter__()
        # air purifier
        ap_pts = [model.add_point((ap_center[0] - ap_l[0] / 2, ap_center[1] - ap_l[1] / 2, 0), mesh_size=resolution),
                  model.add_point(
            (ap_center[0] + ap_l[0] / 2, ap_center[1] - ap_l[1] / 2, 0), mesh_size=resolution),
            model.add_point(
            (ap_center[0] + ap_l[0] / 2, ap_center[1] + ap_l[1] / 2, 0), mesh_size=resolution),
            model.add_point((ap_center[0] - ap_l[0] / 2, ap_center[1] + ap_l[1] / 2, 0), mesh_size=resolution)]
        ap_lines = [model.add_line(ap_pts[i], ap_pts[i + 1])
                    for i in range(-1, len(ap_pts) - 1)]
        # ap_pts2 = [model.add_point((ap_center2[0] - ap_l[0]/2, ap_center2[1] - ap_l[1]/2, 0), mesh_size=resolution),
        #            model.add_point((ap_center2[0] + ap_l[0]/2, ap_center2[1] - ap_l[1]/2, 0), mesh_size=resolution),
        #            model.add_point((ap_center2[0] + ap_l[0]/2, ap_center2[1] + ap_l[1]/2, 0), mesh_size=resolution),
        #            model.add_point((ap_center2[0] - ap_l[0]/2, ap_center2[1] + ap_l[1]/2, 0), mesh_size=resolution)]
        # ap_lines2 = [model.add_line(ap_pts2[i], ap_pts2[i+1])
        #                  for i in range(-1, len(ap_pts2)-1)]
        # print(ap_lines[0].points)
        # print(ap_lines[1].points)
        # print(ap_lines[2].points)
        # print(ap_lines[3].points)
        # Add circle
        circle = [model.add_circle(centers[0], r, mesh_size=resolution),
                  model.add_circle(centers[1], r, mesh_size=resolution),
                  model.add_circle(centers[2], r, mesh_size=resolution),
                  model.add_circle(centers[3], r, mesh_size=resolution),
                  model.add_circle(centers[4], r, mesh_size=resolution),
                  model.add_circle(centers[5], r, mesh_size=resolution)]
        # Add points with finer resolution on left side
        points = [model.add_point((0, 0, 0), mesh_size=coarsening * resolution),
                  model.add_point(
                      (dom_dx, 0, 0), mesh_size=coarsening * resolution),
                  model.add_point((dom_dx, dom_dy / 2 - W_vent / 2, 0),
                                  mesh_size=coarsening * resolution),
                  model.add_point((dom_dx, dom_dy / 2 + W_vent / 2, 0),
                                  mesh_size=coarsening * resolution),
                  model.add_point((dom_dx, dom_dy, 0),
                                  mesh_size=coarsening * resolution),
                  model.add_point(
                      (0, dom_dy, 0), mesh_size=coarsening * resolution),
                  model.add_point((0, dom_dy / 2 + W_vent / 2, 0),
                                  mesh_size=coarsening * resolution),
                  model.add_point((-p_in, dom_dy / 2, 0),
                                  mesh_size=coarsening * resolution),
                  model.add_point((0, dom_dy / 2 - W_vent / 2, 0), mesh_size=coarsening * resolution)]

        # Add lines between all points creating the rectangle
        channel_lines = [model.add_line(points[i], points[i + 1])
                         for i in range(-1, len(points) - 3)]
        channel_lines.append(model.add_circle_arc(
            points[-3], points[-2], points[-1]))
        # inlet_arc = []

        # Create a line loop and plane surface for meshing
        channel_loop = model.add_curve_loop(channel_lines)
        airpurifier = model.add_curve_loop(ap_lines)
        plane_surface = model.add_plane_surface(
            channel_loop, holes=[circle[0].curve_loop,
                                 circle[1].curve_loop,
                                 circle[2].curve_loop,
                                 circle[3].curve_loop,
                                 circle[4].curve_loop,
                                 circle[5].curve_loop, airpurifier])

        # Call gmsh kernel before add physical entities
        model.synchronize()

        volume_marker = 6
        model.add_physical([plane_surface], "Volume")
        model.add_physical([channel_lines[7]], "Inflow")
        model.add_physical([channel_lines[3]], "Outflow")
        model.add_physical([channel_lines[1]], "LowerWall")
        model.add_physical([channel_lines[5]], "UpperWall")
        model.add_physical([channel_lines[0], channel_lines[2],
                            channel_lines[4], channel_lines[6]],  "Walls")
        model.add_physical([ap_lines[0]], "unit1x0")
        model.add_physical([ap_lines[1]], "unit1y0")
        model.add_physical([ap_lines[2]], "unit1x1")
        model.add_physical([ap_lines[3]], "unit1y1")
        # model.add_physical([ap_lines2[0]], "unit2x0")
        # model.add_physical([ap_lines2[1]], "unit2y0")
        # model.add_physical([ap_lines2[2]], "unit2x1")
        # model.add_physical([ap_lines2[3]], "unit2y1")

        model.add_physical(circle[0].curve_loop.curves, "P1")
        model.add_physical(circle[1].curve_loop.curves, "P2")
        model.add_physical(circle[2].curve_loop.curves, "P3")
        model.add_physical(circle[3].curve_loop.curves, "P4")
        model.add_physical(circle[4].curve_loop.curves, "P5")
        model.add_physical(circle[5].curve_loop.curves, "P6")
        # print("LowerWall", channel_lines[1].points)
        # print("UpperWall", channel_lines[5].points)
        geometry.generate_mesh(dim=2)
        pygmsh.write(self.meshPath)
        # gmsh.write(self.meshPath)
        # gmsh.clear()
        geometry.__exit__()


class RemoveBadPlacement_AP2(RemoveBadPlacement):
    def _do(self, problem, pop, **kwargs):
        spacing = 0.3

        def is_overlapping(x):
            x_overlap, y_overlap = True, True
            if abs(x[2] - x[0]) >= spacing + 0.1:
                x_overlap = False
            if abs(x[3] - x[1]) >= spacing + 0.1:
                y_overlap = False
            return bool(x_overlap and y_overlap)
        # move ap1 away from stationary obstacles
        pop = super()._do(problem, pop, **kwargs)
        X = pop.get('X')
        for ind, x in enumerate(X):
            x[0], x[1] = self.moveAP(x[0], x[1])
            x[2], x[3] = self.moveAP(x[2], x[3])
            overlapping = is_overlapping(x)
            while overlapping:
                print('enter while loop')
                options = [spacing, -spacing]
                # x_coor
                if abs(x[2] - x[0]) < spacing:
                    dx = random.choice(options)
                    x[2] += dx
                if abs(x[3] - x[1]) < spacing:
                    dy = random.choice(options)
                    x[3] += dy
                x[2], x[3] = self.moveAP(x[2], x[3])
                print('AP1:', x[0], x[1])
                print('AP2:', x[2], x[3])
                overlapping = is_overlapping(x)
            X[ind] = x
            print('COMPLETED INDIVIDUAL')
            print('AP1:', x[0], x[1])
            print('AP2:', x[2], x[3])
            print()
        pop.set('X', X)
        return pop


class Room2D_2AP(Room2D_AP):
    n_var = 7
    var_labels = ['AP1 x-Location [m]', 'AP1 y-Location [m]',
                  'AP2 x-location [m]', 'AP2 y-location [m]',
                  'AP1 Direction', 'AP2 Direction',
                  'AP ACH [m^2/hr]']
    var_type = ['real', 'real',
                'real', 'real',
                'int', 'int',
                'real']
    xl = [0.5, 0.5, 0.5, 0.5, 1, 1, 0.5]
    xu = [3.5, 3.5, 3.5, 3.5, 4, 4, 6]

    # n_obj = 2
    # obj_labels = ['Room ACH [m^2/hr]', 'Subjects Mean Exposure']

    # n_constr = 0

    repair = RemoveBadPlacement_AP2()

    # externalSolver = True
    procLim = 70
    nProc = 10
    solverExecCmd = ['mpirun', '-n', str(nProc), '2D_room']
    # solverExecCmd = ['sbatch', '--wait', 'jobslurm.sh']
    # onlyParallelizeSolve = True
    # nTasks = 7

    def _preProc(self):
        ap_ach = self.x[6]
        ap_l = 0.3
        room_area = 16  # m^2
        ap_speed = ap_ach * room_area / 3600 / ap_l / 2

        def ap_preProc(direct, unit):
            if direct == 1:
                u_str = f'0. {ap_speed} 0.'
                walls = 'x'
                inlets = 'y'
            elif direct == 2:
                u_str = f'{ap_speed} 0. 0.'
                walls = 'y'
                inlets = 'x'
            elif direct == 3:
                u_str = f'0. -{ap_speed} 0.'
                walls = 'x'
                inlets = 'y'
            elif direct == 4:
                u_str = f'-{ap_speed} 0. 0.'
                walls = 'y'
                inlets = 'x'
            else:
                raise Exception(f'Can\'t handle AP direction parameter - {direct}')
            in_lines = self.input_lines_rw
            bnd_wall_ids = [f'unit{unit}{walls}{surf}' for surf in range(2)]
            bnd_inlet_ids = [f'unit{unit}{inlets}{surf}' for surf in range(2)]
            for id in bnd_wall_ids:
                kw_lines = self.findKeywordLines(id, in_lines)
                for line_i, _ in kw_lines:
                    del in_lines[line_i]
                bnd_lines = [f"BOUNDARY {id} DESCRIPTION = 'ap-{id}'",
                             f'BOUNDARY {id} TYPE = WALL']
                in_lines.extend(bnd_lines)

            for id in bnd_inlet_ids:
                kw_lines = self.findKeywordLines(id, in_lines)
                for line_i, _ in kw_lines:
                    del in_lines[line_i]
                bnd_lines = [f"BOUNDARY {id} DESCRIPTION = 'ap-{id}'",
                             f'BOUNDARY {id} TYPE = INLET',
                             f'BOUNDARY {id} U = ' + u_str,
                             f'BOUNDARY {id} Z = 0.0',
                             f'BOUNDARY {id} PHI_NORMAL_FLUX = 0.0'
                             ]
                in_lines.extend(bnd_lines)
            self.input_lines_rw = in_lines
        self.genMesh()
        ap1_direct = self.x[4]
        ap2_direct = self.x[5]  # 0.0296
        ap_preProc(ap1_direct, 1)
        ap_preProc(ap2_direct, 2)

    def _postProc(self):
        with h5py.File(self.datPath, 'r') as f:
            t = f['Datas']['TOTAL_TIME'][:]
            SP1 = f['Datas']['Surf_Int1'][:]
            SP2 = f['Datas']['Surf_Int2'][:]
            SP3 = f['Datas']['Surf_Int3'][:]
            SP4 = f['Datas']['Surf_Int4'][:]
            SP5 = f['Datas']['Surf_Int5'][:]
            SP6 = f['Datas']['Surf_Int6'][:]
        # tot_dt = t[-1] - t[0]
        SPs = [SP1, SP2, SP3, SP4, SP5, SP6]
        t_start = 1800
        mask = np.where(t > t_start)
        t = t[mask]
        # dt = t[-1] - t_start
        SPs = [sp[mask] for sp in SPs]
        t_wghts = [t[i] - t[i - 1] for i in range(1, len(t))]
        t_mid = [np.mean([t[i], t[i - 1]]) for i in range(1, len(t))]
        emit_person = 3
        t_avgs = []
        for j, sp in enumerate(SPs):
            p = j + 1
            if not p == emit_person:
                plt.plot(t, sp)
                plt.title(f'Passive Scalar Surface Average - Person {p}')
                plt.xlabel('Time [sec]')
                plt.ylabel('Passive Scalar Surface Average [unitless]')
                path = os.path.join(self.abs_path, f'sp-p{p}.png')
                plt.savefig(path)
                plt.clf()
                sp_mid = [np.mean(np.abs([sp[i], sp[i - 1]]))
                          for i in range(1, len(sp))]
                plt.plot(t_mid, sp_mid)
                plt.title(
                    f'Passive Scalar Surface Average - Person {p}: mid-points')
                plt.xlabel('Time [sec]')
                plt.ylabel('Passive Scalar Surface Average [unitless]')
                path = os.path.join(self.abs_path, f'sp_mid-p{p}.png')
                plt.savefig(path)
                plt.clf()
                # plt.show()
                t_avgs.append(np.average(sp_mid, weights=t_wghts))
        saveTxt(self.abs_path, 'surf-avg-scalar.txt', t_avgs)
        tot_avg = np.mean(t_avgs)
        ACH = 2 + self.x[6]
        self.f = [ACH, tot_avg]
        return self.f

    def _genMesh(self):
        resolution = 0.01
        coarsening = 2
        # Channel parameters
        dom_dx = 4
        dom_dy = 4
        W_vent = 0.5
        p_in = 0.125
        # c = [1, 1, 0]
        centers = [[1.25, 1, 0], [2.75, 1, 0],
                   [1.25, 2, 0], [2.75, 2, 0],
                   [1.25, 3, 0], [2.75, 3, 0]]
        r = 0.125
        inlet_c = [-0.1, dom_dy / 2, 0]

        ap_l = [0.3, 0.3]
        ap_center1 = [self.x[0], self.x[1]]
        ap_center2 = [self.x[2], self.x[3]]
        # Initialize empty geometry using the build in kernel in GMSH
        geometry = pygmsh.geo.Geometry()
        # Fetch model we would like to add data to
        model = geometry.__enter__()
        # air purifier
        ap_pts1 = [model.add_point((ap_center1[0] - ap_l[0] / 2, ap_center1[1] - ap_l[1] / 2, 0), mesh_size=resolution),
                   model.add_point(
                       (ap_center1[0] + ap_l[0] / 2, ap_center1[1] - ap_l[1] / 2, 0), mesh_size=resolution),
                   model.add_point(
                       (ap_center1[0] + ap_l[0] / 2, ap_center1[1] + ap_l[1] / 2, 0), mesh_size=resolution),
                   model.add_point((ap_center1[0] - ap_l[0] / 2, ap_center1[1] + ap_l[1] / 2, 0), mesh_size=resolution)]
        ap_lines1 = [model.add_line(ap_pts1[i], ap_pts1[i + 1])
                     for i in range(-1, len(ap_pts1) - 1)]
        ap_pts2 = [model.add_point((ap_center2[0] - ap_l[0] / 2, ap_center2[1] - ap_l[1] / 2, 0), mesh_size=resolution),
                   model.add_point(
                       (ap_center2[0] + ap_l[0] / 2, ap_center2[1] - ap_l[1] / 2, 0), mesh_size=resolution),
                   model.add_point(
                       (ap_center2[0] + ap_l[0] / 2, ap_center2[1] + ap_l[1] / 2, 0), mesh_size=resolution),
                   model.add_point((ap_center2[0] - ap_l[0] / 2, ap_center2[1] + ap_l[1] / 2, 0), mesh_size=resolution)]
        ap_lines2 = [model.add_line(ap_pts2[i], ap_pts2[i + 1])
                     for i in range(-1, len(ap_pts2) - 1)]
        # print(ap_lines[0].points)
        # print(ap_lines[1].points)
        # print(ap_lines[2].points)
        # print(ap_lines[3].points)
        # Add circle
        circle = [model.add_circle(centers[0], r, mesh_size=resolution),
                  model.add_circle(centers[1], r, mesh_size=resolution),
                  model.add_circle(centers[2], r, mesh_size=resolution),
                  model.add_circle(centers[3], r, mesh_size=resolution),
                  model.add_circle(centers[4], r, mesh_size=resolution),
                  model.add_circle(centers[5], r, mesh_size=resolution)]
        # Add points with finer resolution on left side
        points = [model.add_point((0, 0, 0), mesh_size=coarsening * resolution),
                  model.add_point(
                      (dom_dx, 0, 0), mesh_size=coarsening * resolution),
                  model.add_point((dom_dx, dom_dy / 2 - W_vent / 2, 0),
                                  mesh_size=coarsening * resolution),
                  model.add_point((dom_dx, dom_dy / 2 + W_vent / 2, 0),
                                  mesh_size=coarsening * resolution),
                  model.add_point((dom_dx, dom_dy, 0),
                                  mesh_size=coarsening * resolution),
                  model.add_point(
                      (0, dom_dy, 0), mesh_size=coarsening * resolution),
                  model.add_point((0, dom_dy / 2 + W_vent / 2, 0),
                                  mesh_size=coarsening * resolution),
                  model.add_point((-p_in, dom_dy / 2, 0),
                                  mesh_size=coarsening * resolution),
                  model.add_point((0, dom_dy / 2 - W_vent / 2, 0), mesh_size=coarsening * resolution)]

        # Add lines between all points creating the rectangle
        channel_lines = [model.add_line(points[i], points[i + 1])
                         for i in range(-1, len(points) - 3)]
        channel_lines.append(model.add_circle_arc(
            points[-3], points[-2], points[-1]))
        # inlet_arc = []

        # Create a line loop and plane surface for meshing
        channel_loop = model.add_curve_loop(channel_lines)
        airpurifier1 = model.add_curve_loop(ap_lines1)
        airpurifier2 = model.add_curve_loop(ap_lines2)
        plane_surface = model.add_plane_surface(
            channel_loop, holes=[circle[0].curve_loop,
                                 circle[1].curve_loop,
                                 circle[2].curve_loop,
                                 circle[3].curve_loop,
                                 circle[4].curve_loop,
                                 circle[5].curve_loop, airpurifier1, airpurifier2])

        # Call gmsh kernel before add physical entities
        model.synchronize()

        volume_marker = 6
        model.add_physical([plane_surface], "Volume")
        model.add_physical([channel_lines[7]], "Inflow")
        model.add_physical([channel_lines[3]], "Outflow")
        model.add_physical([channel_lines[1]], "LowerWall")
        model.add_physical([channel_lines[5]], "UpperWall")
        model.add_physical([channel_lines[0], channel_lines[2],
                            channel_lines[4], channel_lines[6]],  "Walls")
        model.add_physical([ap_lines1[0]], "unit1x0")
        model.add_physical([ap_lines1[1]], "unit1y0")
        model.add_physical([ap_lines1[2]], "unit1x1")
        model.add_physical([ap_lines1[3]], "unit1y1")
        model.add_physical([ap_lines2[0]], "unit2x0")
        model.add_physical([ap_lines2[1]], "unit2y0")
        model.add_physical([ap_lines2[2]], "unit2x1")
        model.add_physical([ap_lines2[3]], "unit2y1")

        model.add_physical(circle[0].curve_loop.curves, "P1")
        model.add_physical(circle[1].curve_loop.curves, "P2")
        model.add_physical(circle[2].curve_loop.curves, "P3")
        model.add_physical(circle[3].curve_loop.curves, "P4")
        model.add_physical(circle[4].curve_loop.curves, "P5")
        model.add_physical(circle[5].curve_loop.curves, "P6")
        # print("LowerWall", channel_lines[1].points)
        # print("UpperWall", channel_lines[5].points)
        geometry.generate_mesh(dim=2)
        pygmsh.write(self.meshPath)
        geometry.__exit__()
