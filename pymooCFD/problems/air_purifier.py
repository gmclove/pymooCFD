from pymooCFD.core.cfdCase import YALES2Case

import h5py
import os
import pandas as pd
import pygmsh
import gmsh
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.repair import Repair


class RemoveBadPlacement(Repair):
    def _do(self, problem, pop, **kwargs):
        # Channel parameters
        dom_x0, dom_y0, dom_z0 = 0, 0, 0
        dom_dx, dom_dy = 4, 4
        # W_vent = 0.5
        # p_in = 0.125
        # c = [1, 1, 0]
        centers = [[1.25, 1, 0], [2.75, 1, 0],
                   [1.25, 2, 0], [2.75, 2, 0],
                   [1.25, 3, 0], [2.75, 3, 0]]
        r = 0.125
        # inlet_c = [-0.1, dom_dy/2, 0]

        ap_l = [0.3, 0.3]
        # ap_center1 = [2, 3.5]
        # ap_center2 = [2, 0.5]

        spacing = 0.2
        wall_space_dx = ap_l[0] + spacing
        wall_space_dy = ap_l[1] + spacing
        min_x, min_y = dom_x0 + wall_space_dx, dom_y0 + wall_space_dy
        max_x, max_y = dom_dx + dom_x0 - wall_space_dx, dom_y0 + dom_dy - wall_space_dy
        person_space = r + spacing

        rm_inds = []
        for i in range(len(pop)):
            x = pop[i].X
            x_coor, y_coor = x[0], x[1]
            if x_coor < min_x:
                rm_inds.append(i)
                continue
            if x_coor > max_x:
                rm_inds.append(i)
                continue
            if y_coor < min_y:
                rm_inds.append(i)
                continue
            if y_coor > max_y:
                rm_inds.append(i)
                continue
            for cir_cent in centers:
                cir_cx = cir_cent[0]
                cir_cy = cir_cent[1]
                x_max, x_min = cir_cx + person_space, cir_cx - person_space
                y_max, y_min = cir_cy + person_space, cir_cy - person_space
                if x_coor < x_max and x_coor > x_min:
                    rm_inds.append(i)
                    break
                if y_coor < y_max and y_coor > y_min:
                    rm_inds.append(i)
                    break
        X = pop.get('X')
        X = [x for i, x in enumerate(X) if i not in rm_inds]
        pop.set('X', X)
        return pop


class Room2D_AP(YALES2Case):
    n_var = 2
    var_labels = ['x-location', 'y-location']
    var_type = ['real', 'real']
    xl = [0.5, 0.5]
    xu = [3.5, 3.5]

    n_obj = 2
    obj_labels = ['ACH', 'Mean Exposure']

    repair = RemoveBadPlacement()

    def __init__(self, case_path, x, meshSF=1, **kwargs):
        super().__init__(case_path, x, meshSF=meshSF,
                         meshFile='room_6P_ap.msh22',
                         datFile=os.path.join('dump', 'temporal.h5'),
                         **kwargs)

    # def _preProc(self):
    #     pass

    def _postProc(self):
        with h5py.File(self.datPath, 'r') as f:
            print(f['Datas'].keys())
            print(f['Datas']['TIME'][:])
            print(f['Datas']['TOTAL_TIME'][:])
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
        t_wghts = [t[i] - t[i-1] for i in range(1, len(t))]
        t_mid = [np.mean([t[i], t[i-1]]) for i in range(1, len(t))]
        t_avgs = []
        for j, sp in enumerate(SPs):
            p = j+1
            if not p == 3:
                plt.plot(t, sp)
                plt.title(f'Passive Scalar Surface Average - Person {p}')
                plt.xlabel('Time [sec]')
                plt.ylabel('Passive Scalar Surface Average [unitless]')
                plt.savefig(f'sp-p{p}.png')
                plt.clf()
                sp_mid = [np.mean(np.abs([sp[i], sp[i-1]])) for i in range(1, len(sp))]
                plt.plot(t_mid, sp_mid)
                plt.title(f'Passive Scalar Surface Average - Person {p}: mid-points')
                plt.xlabel('Time [sec]')
                plt.ylabel('Passive Scalar Surface Average [unitless]')
                plt.savefig(f'sp_mid-p{p}.png')
                plt.clf()
                # plt.show()
                t_avgs.append(np.average(sp_mid, weights=t_wghts))
        tot_avg = np.mean(t_avgs)


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
        print("LowerWall", channel_lines[1].points)
        print("UpperWall", channel_lines[5].points)
        geometry.generate_mesh(dim=2)
        pygmsh.write(self.meshPath)
        # gmsh.write(self.meshPath)
        # gmsh.clear()
        geometry.__exit__()


# class Room2D_AP2(Room2D_AP):
#     def _genMesh(self):
#         resolution = 0.01
#         coarsening = 2
#         # Channel parameters
#         dom_dx = 4
#         dom_dy = 4
#         W_vent = 0.5
#         p_in = 0.125
#         # c = [1, 1, 0]
#         centers = [[1.25, 1, 0], [2.75, 1, 0],
#                    [1.25, 2, 0], [2.75, 2, 0],
#                    [1.25, 3, 0], [2.75, 3, 0]]
#         r = 0.125
#         inlet_c = [-0.1, dom_dy / 2, 0]
#
#         ap_l = [0.3, 0.3]
#         ap_center1 = [2, 3.5]
#         ap_center2 = [2, 0.5]
#         # Initialize empty geometry using the build in kernel in GMSH
#         geometry = pygmsh.geo.Geometry()
#         # Fetch model we would like to add data to
#         model = geometry.__enter__()
#         # air purifier
#         ap_pts1 = [model.add_point((ap_center1[0] - ap_l[0] / 2, ap_center1[1] - ap_l[1] / 2, 0), mesh_size=resolution),
#                    model.add_point(
#                        (ap_center1[0] + ap_l[0] / 2, ap_center1[1] - ap_l[1] / 2, 0), mesh_size=resolution),
#                    model.add_point(
#                        (ap_center1[0] + ap_l[0] / 2, ap_center1[1] + ap_l[1] / 2, 0), mesh_size=resolution),
#                    model.add_point((ap_center1[0] - ap_l[0] / 2, ap_center1[1] + ap_l[1] / 2, 0), mesh_size=resolution)]
#         ap_lines1 = [model.add_line(ap_pts1[i], ap_pts1[i + 1])
#                      for i in range(-1, len(ap_pts1) - 1)]
#         ap_pts2 = [model.add_point((ap_center2[0] - ap_l[0] / 2, ap_center2[1] - ap_l[1] / 2, 0), mesh_size=resolution),
#                    model.add_point(
#                        (ap_center2[0] + ap_l[0] / 2, ap_center2[1] - ap_l[1] / 2, 0), mesh_size=resolution),
#                    model.add_point(
#                        (ap_center2[0] + ap_l[0] / 2, ap_center2[1] + ap_l[1] / 2, 0), mesh_size=resolution),
#                    model.add_point((ap_center2[0] - ap_l[0] / 2, ap_center2[1] + ap_l[1] / 2, 0), mesh_size=resolution)]
#         ap_lines2 = [model.add_line(ap_pts2[i], ap_pts2[i + 1])
#                      for i in range(-1, len(ap_pts2) - 1)]
#         # print(ap_lines[0].points)
#         # print(ap_lines[1].points)
#         # print(ap_lines[2].points)
#         # print(ap_lines[3].points)
#         # Add circle
#         circle = [model.add_circle(centers[0], r, mesh_size=resolution),
#                   model.add_circle(centers[1], r, mesh_size=resolution),
#                   model.add_circle(centers[2], r, mesh_size=resolution),
#                   model.add_circle(centers[3], r, mesh_size=resolution),
#                   model.add_circle(centers[4], r, mesh_size=resolution),
#                   model.add_circle(centers[5], r, mesh_size=resolution)]
#         # Add points with finer resolution on left side
#         points = [model.add_point((0, 0, 0), mesh_size=coarsening * resolution),
#                   model.add_point(
#                       (dom_dx, 0, 0), mesh_size=coarsening * resolution),
#                   model.add_point((dom_dx, dom_dy / 2 - W_vent / 2, 0),
#                                   mesh_size=coarsening * resolution),
#                   model.add_point((dom_dx, dom_dy / 2 + W_vent / 2, 0),
#                                   mesh_size=coarsening * resolution),
#                   model.add_point((dom_dx, dom_dy, 0),
#                                   mesh_size=coarsening * resolution),
#                   model.add_point(
#                       (0, dom_dy, 0), mesh_size=coarsening * resolution),
#                   model.add_point((0, dom_dy / 2 + W_vent / 2, 0),
#                                   mesh_size=coarsening * resolution),
#                   model.add_point((-p_in, dom_dy / 2, 0),
#                                   mesh_size=coarsening * resolution),
#                   model.add_point((0, dom_dy / 2 - W_vent / 2, 0), mesh_size=coarsening * resolution)]
#
#         # Add lines between all points creating the rectangle
#         channel_lines = [model.add_line(points[i], points[i + 1])
#                          for i in range(-1, len(points) - 3)]
#         channel_lines.append(model.add_circle_arc(
#             points[-3], points[-2], points[-1]))
#         # inlet_arc = []
#
#         # Create a line loop and plane surface for meshing
#         channel_loop = model.add_curve_loop(channel_lines)
#         airpurifier1 = model.add_curve_loop(ap_lines1)
#         airpurifier2 = model.add_curve_loop(ap_lines2)
#         plane_surface = model.add_plane_surface(
#             channel_loop, holes=[circle[0].curve_loop,
#                                  circle[1].curve_loop,
#                                  circle[2].curve_loop,
#                                  circle[3].curve_loop,
#                                  circle[4].curve_loop,
#                                  circle[5].curve_loop, airpurifier1, airpurifier2])
#
#         # Call gmsh kernel before add physical entities
#         model.synchronize()
#
#         volume_marker = 6
#         model.add_physical([plane_surface], "Volume")
#         model.add_physical([channel_lines[7]], "Inflow")
#         model.add_physical([channel_lines[3]], "Outflow")
#         model.add_physical([channel_lines[1]], "LowerWall")
#         model.add_physical([channel_lines[5]], "UpperWall")
#         model.add_physical([channel_lines[0], channel_lines[2],
#                             channel_lines[4], channel_lines[6]],  "Walls")
#         model.add_physical([ap_lines1[0]], "unit1x0")
#         model.add_physical([ap_lines1[1]], "unit1y0")
#         model.add_physical([ap_lines1[2]], "unit1x1")
#         model.add_physical([ap_lines1[3]], "unit1y1")
#         model.add_physical([ap_lines2[0]], "unit2x0")
#         model.add_physical([ap_lines2[1]], "unit2y0")
#         model.add_physical([ap_lines2[2]], "unit2x1")
#         model.add_physical([ap_lines2[3]], "unit2y1")
#
#         model.add_physical(circle[0].curve_loop.curves, "P1")
#         model.add_physical(circle[1].curve_loop.curves, "P2")
#         model.add_physical(circle[2].curve_loop.curves, "P3")
#         model.add_physical(circle[3].curve_loop.curves, "P4")
#         model.add_physical(circle[4].curve_loop.curves, "P5")
#         model.add_physical(circle[5].curve_loop.curves, "P6")
#         print("LowerWall", channel_lines[1].points)
#         print("UpperWall", channel_lines[5].points)
#         geometry.generate_mesh(dim=2)
#         import gmsh
#         gmsh.write("room_6P_ap.msh22")
#         gmsh.clear()
#         geometry.__exit__()
