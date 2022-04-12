# @Author: glove
# @Date:   2021-12-10T11:22:40-05:00
# @Last modified by:   glove
# @Last modified time: 2021-12-14T15:45:19-05:00
<<<<<<< HEAD
# from pymoo.visualization.scatter import Scatter
import numpy as np
# import os


dat = np.genfromtxt('residuals.dat')
print(dat)
print(dat[-1, 0])
# if dat[-1, 0] < 2000:
#     raise Exception
avgs = []
for col in dat.T[1:]:
    avgs.append(np.mean(col))
print(np.mean(avgs))
print(np.mean(dat[:, 1:]))

=======

class A:
    a = 1

    def __init__(self):
        self.b = 2
        # self.externalSolver =
inst = A()
Cls = A
print(Cls.__dict__)
print(inst.__dict__)
print(inst.__class__.__dict__)
inst.__class__.__dict__ = inst.__class__.__dict__
print(inst.__class__.__dict__)


# from pymoo.visualization.scatter import Scatter
# import numpy as np
# import os
>>>>>>> e5cab1f77b4578ddaad5cfcd097b366e5f34fd9c
#
# import gmsh
#
# meshSF = 1.0
#
# projName = '2D_cylinder'
# cylD = 1
# cylR = cylD / 2
# cyl_cx, cyl_cy, cyl_cz = 0, 0, 0
# dom_dx, dom_dy, dom_dz = cylD * 60, cylD * 30, 1
# dom_ox, dom_oy, dom_oz = -dom_dx / 6 + cyl_cx, - \
#     dom_dy / 2 + cyl_cy, 0 + cyl_cz  # -dom_dz/2
# meshSizeMin, meshSizeMax = 0.01 * meshSF, 0.4
# #################################
# #          Initialize           #
# #################################
# gmsh.initialize()
# # By default Gmsh will not print out any messages: in order to output messages
# # on the terminal, just set the "General.Terminal" option to 1:
# gmsh.option.setNumber("General.Terminal", 0)
# gmsh.clear()
# gmsh.model.add(projName)
# # gmsh.option.setNumber('Mesh.MeshSizeFactor', meshSF)
# #################################
# #      YALES2 Requirements      #
# #################################
# # Make sure "Recombine all triangular meshes" is unchecked so only triangular elements are produced
# gmsh.option.setNumber('Mesh.RecombineAll', 0)
# # Only save entities that are assigned to a physical group
# gmsh.option.setNumber('Mesh.SaveAll', 0)
# #################################
# #           Geometry            #
# #################################
# rect = gmsh.model.occ.addRectangle(
#     dom_ox, dom_oy, dom_oz, dom_dx, dom_dy)
# # add circle to rectangular domain to represent cylinder
# cir = gmsh.model.occ.addCircle(0, 0, 0, cylR)
# # use 1-D circle to create curve loop entity
# cir_loop = gmsh.model.occ.addCurveLoop([cir])
# cir_plane = gmsh.model.occ.addPlaneSurface(
#     [cir_loop])  # creates 2-D entity
# # cut circle out of a rectangle
# # print(cylLoop)
# # print(rect)
# domDimTags, domDimTagsMap = gmsh.model.occ.cut(
#     [(2, rect)], [(2, cir_plane)])
# # divide domain into 4 regions
# p_top_cyl = gmsh.model.occ.addPoint(0, cyl_cy + cylR, 0)
# p_bot_cyl = gmsh.model.occ.addPoint(0, cyl_cy - cylR, 0)
# p_left_cyl = gmsh.model.occ.addPoint(cyl_cx - cylR, 0, 0)
# p_right_cyl = gmsh.model.occ.addPoint(cyl_cx + cylR, 0, 0)
#
# p_top_dom = gmsh.model.occ.addPoint(0, dom_oy + dom_dy, 0)
# p_bot_dom = gmsh.model.occ.addPoint(0, dom_oy, 0)
# p_left_dom = gmsh.model.occ.addPoint(dom_ox, 0, 0)
# p_right_dom = gmsh.model.occ.addPoint(dom_ox + dom_dx, 0, 0)
#
# l_top = gmsh.model.occ.addLine(p_top_cyl, p_top_dom)
# l_bot = gmsh.model.occ.addLine(p_bot_cyl, p_bot_dom)
# l_left = gmsh.model.occ.addLine(p_left_cyl, p_left_dom)
# l_right = gmsh.model.occ.addLine(p_right_cyl, p_right_dom)
#
# domDimTags, domDimTagsMap = gmsh.model.occ.fragment(
#     domDimTags, [(1, l_top), (1, l_bot), (1, l_left), (1, l_right)])
# # We finish by synchronizing the data from OpenCASCADE CAD kernel with
# # the Gmsh model:
# gmsh.model.occ.synchronize()
# #################################
# #    Physical Group Naming      #
# #################################
# dim = 2
# grpTag = gmsh.model.addPhysicalGroup(dim, range(1, 4 + 1))
# gmsh.model.setPhysicalName(dim, grpTag, 'dom')
#
# dim = 1
# grpTag = gmsh.model.addPhysicalGroup(dim, [14, 20])
# gmsh.model.setPhysicalName(dim, grpTag, 'x0')
# grpTag = gmsh.model.addPhysicalGroup(dim, [25, 18])
# gmsh.model.setPhysicalName(dim, grpTag, 'x1')
# grpTag = gmsh.model.addPhysicalGroup(dim, [16, 19])
# gmsh.model.setPhysicalName(dim, grpTag, 'y0')
# grpTag = gmsh.model.addPhysicalGroup(dim, [21, 24])
# gmsh.model.setPhysicalName(dim, grpTag, 'y1')
# grpTag = gmsh.model.addPhysicalGroup(dim, [15, 17, 22, 23])
# gmsh.model.setPhysicalName(dim, grpTag, 'cyl')
# #################################
# #           MESHING             #
# #################################
# # TRANSFINITE CURVE
# bnds_right = gmsh.model.getParametrizationBounds(1, l_right)
# len_right = abs(bnds_right[1][0] - bnds_right[0][0])
# bnds_left = gmsh.model.getParametrizationBounds(1, l_left)
# len_left = abs(bnds_left[1][0] - bnds_left[0][0])
# bnds_top = gmsh.model.getParametrizationBounds(1, l_top)
# len_top = abs(bnds_top[1][0] - bnds_top[0][0])
# bnds_bot = gmsh.model.getParametrizationBounds(1, l_bot)
# len_bot = abs(bnds_bot[1][0] - bnds_bot[0][0])
#
#
# def get_coeff_and_NN(x_min, x_max, x_tot, NN_init=100, coef_init=1.001):
#     max_it = 40
#     it = 0
#     thresh = 1e-6
#     err = np.inf
#
#     x_0 = x_min
#     x_f = x_max
#     NN = NN_init
#     coef = coef_init
#     while it < max_it and err > thresh:
#         coef_prev = coef
#         NN = int(np.log(1 + x_tot / x_0 * (coef - 1)) / np.log(coef) + 3)
#         coef = np.e**(np.log(x_f / x_0) / (NN - 3))
#         err = abs(coef_prev - coef)
#         it += 1
#         # print(it, err, coef, NN)
#     return coef, NN
#
#
# x_min = meshSizeMin
# x_max = meshSizeMax
# coef_left, NN_left = get_coeff_and_NN(x_min, x_max, len_left)
# coef_right, NN_right = get_coeff_and_NN(x_min, x_max, len_right)
# coef_bot, NN_bot = get_coeff_and_NN(x_min, x_max, len_bot)
# coef_top, NN_top = get_coeff_and_NN(x_min, x_max, len_top)
#
# gmsh.model.mesh.setTransfiniteCurve(l_bot, NN_bot, coef=coef_bot)
# gmsh.model.mesh.setTransfiniteCurve(l_top, NN_top, coef=coef_top)
# gmsh.model.mesh.setTransfiniteCurve(l_left, NN_left, coef=coef_left)
# gmsh.model.mesh.setTransfiniteCurve(l_right, NN_right, coef=coef_right)
#
# len_quarter_cyl = 2 * np.pi * cylR / 4
# NN_quarter_cyl = int(len_quarter_cyl / x_min)
# cyl_tags = [15, 17, 22, 23]
# for tag in cyl_tags:
#     gmsh.model.mesh.setTransfiniteCurve(tag, NN_quarter_cyl)
# # Set minimum and maximum mesh size
# # gmsh.option.setNumber('Mesh.MeshSizeMin', meshSizeMin)
# gmsh.option.setNumber('Mesh.MeshSizeMax', meshSizeMax)
#
# # Set number of nodes along cylinder wall
# # gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 200)
# # gmsh.option.setNumber('Mesh.MeshSizeFromCurvatureIsotropic', 1)
#
# # Set size of mesh at every point in model
# # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshSize)
#
# # gmsh.model.mesh.setTransfiniteCurve(cylCir, 150, coef=1.1)
# # We can then generate a 2D mesh...
# gmsh.model.mesh.generate(1)
# gmsh.model.mesh.generate(2)
# # extract elements
# elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements()
# # count number of elements
# numElem = sum(len(i) for i in elemTags)
# # print('Number of Elements:', numElem)
# ##################
# #    FINALIZE    #
# ##################
# # ... and save it to disk
# # gmsh.write(self.meshPath)
# # To visualize the model we can run the graphical user interface with
# # `gmsh.fltk.run()'.
# gmsh.fltk.run()
# # This should be called when you are done using the Gmsh Python API:
# gmsh.finalize()


#
# import logging
# import matplotlib.pyplot as plt
# plt_logger = logging.getLogger(plt.__name__)
# plt_logger.setLevel(logging.INFO)
# print(plt_logger)
#
# plot_logger = logging.getLogger(Scatter.__name__)
# print(scat_logger)
# plot_logger.setLevel(logging.INFO)
# print(scat_logger)
# plt.set_loglevel("info")


#
# class A:
#     a = 1
#     b = 2
#     c = 3
#
#
# a_obj = [A() for _ in range(4)]
# a = [[obj.a, obj.b] for obj in a_obj]
# print(a)

# def findAndReplaceKeywordLines(file_lines, newLine, kws, replaceOnce=False, exact=False, stripKW=True):
#     def findKeywordLines(kw, file_lines, exact=False):
#         kw_lines = []
#         for line_i, line in enumerate(file_lines):
#             if exact and kw.rstrip().lstrip() == line:
#                 kw_lines.append([line_i, line])
#             elif line.find(kw.rstrip().lstrip()) >= 0:
#                 kw_lines.append([line_i, line])
#         return kw_lines
#     '''
#     Finds and replaces any file_lines with newLine that match keywords (kws) give.
#     If no keyword lines are found the newLine is inserted at the beginning of the file_lines.
#     '''
#     kw_lines_array = []
#     for kw in kws:
#         kw_lines_array.append(self.findKeywordLines(
#             kw, file_lines, exact=exact, stripKW=stripKW))
#     print(kw_lines_array)
#     if sum([len(kw_lines) for kw_lines in kw_lines_array]) > 0:
#         def replace():
#             for kw_lines in kw_lines_array:
#                 for line_i, line in kw_lines:
#                     file_lines[line_i] = newLine
#                     if replaceOnce:
#                         return
#         replace()
#     else:
#         file_lines.insert(0, newLine)
#     return file_lines
#
#
# lines = ['1', '2', 'rggv4tr', 'v45v45']
# kws = ['1    ', '2']
# newLine = 'test'
# new_lines = findAndReplaceKeywordLines(
#     lines, newLine, kws)  # , replaceOnce=True)  # , exact=True)
# print(new_lines)

# kw_lines = []
# for line_i, line in enumerate(file_lines):
#     if exact and kw.rstrip().lstrip() == line:
#         kw_lines.append([line_i, line])
#     elif line.find(kw) >= 0:
#         kw_lines.append([line_i, line])
# return kw_lines


# newLine = f'#SBATCH --cpus-per-task={c}'
# kw_lines_1 = self.findKeywordLines(
#     '#SBATCH --cpus-per-task', job_lines)
# kw_lines_2 = self.findKeywordLines('#SBATCH -c', job_lines)
# if len(kw_lines_1) > 0 or len(kw_lines_2) > 0:
#     for line_i, line in kw_lines_1:
#         job_lines[line_i] = newLine
#     for line_i, line in kw_lines_2:
#         job_lines[line_i] = newLine

# l = ['0', '1', '2', '3']
# print(l)
# l.insert(1, 'new-1')
# print(l)
# l.insert(2, 'new-2')
# print(l)

# path = None
#
# with open(path, 'r') as f:
#     lines = f.readlines()

# val1 = None
# val2 = 2
# if val1 is None or val2 is None:
#     print('is None')

# def fun():
#     return val
# print(fun())

# sentence = "The cat is brown"
# q = "  The cat is brown   "
# print(q.rstrip().lstrip())
# if q.rstrip().lstrip() == sentence:
#     print('strings equal')

# l = [['val1', 2], ['val2', 3]]
# for x, y in l:
#     print(x)
#     print(y)


# path = os.path.join('.', 'case.npy')
# new_path = path.replace('.npy', '')
# print(path)
# print(new_path)
# print(new_path+'.npy')
# val = 1.2345
# a = np.array(val)
# print(a)
# print(a.shape)
# print(bool(a.shape))
# b= np.array([1,2])
# print(bool(b.shape))
# if a.shape:
#     print('a',a.shape)
# if b.shape:
#     print('b', b.shape)
#
# if a.shape is None:
#     print('a.shape is None')
#
# if a.shape[0] is None:
#     print('a.shape[0] is None')

# l = [4 , 3, 2 , 1]
# new_l = np.delete(l, 1)
# print(new_l)


# from setupOpt import optStudy
#
# print(optStudy.algorithm.has_next())
# print(optStudy.algorithm.termination)
# print(optStudy.algorithm.termination.n_gen)
# optStudy.plotBndPts(optStudy.getBndPts())


# s = 'blank'
# assert not isinstance(s, str)

# import time
# import multiprocessing.pool
# pool = multiprocessing.pool.ThreadPool(8)
#
# def wait():
#     print('waiting')
#     time.sleep(5)
#     print('done waiting')
#
# for _ in range(10):
#     pool.apply_async(wait, ())
# pool.close()
# pool.join()

# import gmsh
# import math
#
# meshPath = './test.msh22'
# meshSF = 0.5
#
# mouthD = 0.02
#
# projName = 'jet_cone_axi-sym'
#
# cyl_r = mouthD / 2  # 0.02/2
#
# sph_r = (0.15 / 0.02) * cyl_r
# sph_xc = -sph_r
# sph_yc = 0
#
# cone_dx = 3.5
# cone_r1 = 0.5 / 2
# cone_r2 = cone_dx * (0.7 / 2.5) + cone_r1
# cone_xc = -0.5
#
# # Meshing Scheme
# NN_inlet = int((20 / 0.01) * cyl_r)
# sph_wall_len = sph_r * math.pi - \
#     math.asin(cyl_r / (2 * math.pi)) * sph_r
# NN_sph_wall = int((100 / 0.2355) * sph_wall_len)
# NN_axis_r = 500
# axis_l_len = (sph_xc - sph_r) + 0.5
# NN_axis_l = int((40 / 0.35) * axis_l_len)
# NN_cyl_axis = int((70 / 0.075) * sph_r)
# #################################
# #          Initialize           #
# #################################
# gmsh.initialize()
# gmsh.model.add(projName)
# gmsh.option.setNumber('General.Terminal', 0)
# gmsh.option.setNumber('Mesh.MeshSizeFactor', meshSF)
# #################################
# #      YALES2 Requirements      #
# #################################
# # Make sure "Recombine all triangular meshes" is unchecked so only triangular elements are produced
# gmsh.option.setNumber('Mesh.RecombineAll', 0)
# # Only save entities that are assigned to a physical group
# gmsh.option.setNumber('Mesh.SaveAll', 0)
# #################################
# #           Geometry            #
# #################################
# ##### Points #####
# # Cone
# cone_ll = gmsh.model.occ.addPoint(cone_xc, 0, 0)
# cone_lr = gmsh.model.occ.addPoint(cone_dx + cone_xc, 0, 0)
# cone_ur = gmsh.model.occ.addPoint(cone_dx + cone_xc, cone_r2, 0)
# cone_ul = gmsh.model.occ.addPoint(cone_xc, cone_r1, 0)
# # sphere
# sph_start = gmsh.model.occ.addPoint(sph_xc - sph_r, sph_yc, 0)
# sph_cent = gmsh.model.occ.addPoint(sph_xc, sph_yc, 0)
# sph_end_x = sph_xc + math.sqrt(sph_r**2 - cyl_r**2)
# sph_end = gmsh.model.occ.addPoint(sph_end_x, sph_yc + cyl_r, 0)
# # cylinder
# cyl_ll = sph_cent
# cyl_ul = gmsh.model.occ.addPoint(sph_xc, sph_yc + cyl_r, 0)
# cyl_ur = sph_end
# cyl_mid_outlet = gmsh.model.occ.addPoint(sph_xc + sph_r, 0, 0)
# ##### Curves #####
# # sphere wall
# sph_wall = gmsh.model.occ.addCircleArc(sph_start, sph_cent, sph_end)
# # cylinder wall
# cyl_wall = gmsh.model.occ.addLine(cyl_ul, cyl_ur)
# # cylinder axis (for meshing control)
# axis_cyl = cyl_axis = gmsh.model.occ.addLine(cyl_ll, cyl_mid_outlet)
# # inlet (cylinder back wall)
# inlet = gmsh.model.occ.addLine(cyl_ul, cyl_ll)
# # axis
# axis_l = gmsh.model.occ.addLine(sph_start, cone_ll)
# axis_r = gmsh.model.occ.addLine(cyl_mid_outlet, cone_lr)
# axis = [axis_l, axis_cyl, axis_r]
# # cone
# back_wall = gmsh.model.occ.addLine(cone_ll, cone_ul)
# cone_wall = gmsh.model.occ.addLine(cone_ul, cone_ur)
# outlet = gmsh.model.occ.addLine(cone_ur, cone_lr)
# # mesh field line
# # field_line = gmsh.model.occ.addLine(sph_end, cone_ur)
#
# ##### Surfaces #####
# # domain surface
# curv_loop_tags = [sph_wall, cyl_wall, inlet, axis_cyl, axis_r, outlet,
#                   cone_wall, back_wall, axis_l]
# dom_loop = gmsh.model.occ.addCurveLoop(curv_loop_tags)
# dom = gmsh.model.occ.addPlaneSurface([dom_loop])
#
# gmsh.model.occ.synchronize()
# #################################
# #           MESHING             #
# #################################
# gmsh.option.setNumber('Mesh.FlexibleTransfinite', 1)
# gmsh.model.mesh.setTransfiniteCurve(sph_wall, NN_sph_wall)
# gmsh.model.mesh.setTransfiniteCurve(cyl_wall, NN_cyl_axis)
# gmsh.model.mesh.setTransfiniteCurve(cyl_axis, NN_cyl_axis)
# gmsh.model.mesh.setTransfiniteCurve(inlet, NN_inlet)
# gmsh.model.mesh.setTransfiniteCurve(
#     axis_r, NN_axis_r, meshType='Progression', coef=1.005)
# gmsh.model.mesh.setTransfiniteCurve(axis_l, NN_axis_l)
# gmsh.model.mesh.setTransfiniteCurve(back_wall, 15)
# gmsh.model.mesh.setTransfiniteCurve(
#     outlet, 50, meshType='Progression', coef=0.99)
# gmsh.model.mesh.setTransfiniteCurve(cone_wall, 100)
# ##### Execute Meshing ######
# gmsh.model.mesh.generate(1)
# gmsh.model.mesh.generate(2)
# # extract number of elements
# # get all elementary entities in the model
# entities = gmsh.model.getEntities()
# e = entities[-1]
# # get the mesh elements for each elementary entity
# elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(
#     e[0], e[1])
# # count number of elements
# numElem = sum(len(i) for i in elemTags)
# print(numElem)
# #################################
# #    Physical Group Naming      #
# #################################
# # 1-D physical groups
# dim = 1
# # inlet
# grpTag = gmsh.model.addPhysicalGroup(dim, [inlet])
# gmsh.model.setPhysicalName(dim, grpTag, 'inlet')
# # outlet
# grpTag = gmsh.model.addPhysicalGroup(dim, [outlet])
# gmsh.model.setPhysicalName(dim, grpTag, 'outlet')
# # axis of symmetry
# grpTag = gmsh.model.addPhysicalGroup(dim, axis)
# gmsh.model.setPhysicalName(dim, grpTag, 'axis')
# # walls
# grpTag = gmsh.model.addPhysicalGroup(dim, [sph_wall, cyl_wall])
# gmsh.model.setPhysicalName(dim, grpTag, 'walls')
# # coflow
# grpTag = gmsh.model.addPhysicalGroup(dim, [back_wall, cone_wall])
# gmsh.model.setPhysicalName(dim, grpTag, 'coflow')
#
# # 2-D physical groups
# dim = 2
# grpTag = gmsh.model.addPhysicalGroup(dim, [dom])
# gmsh.model.setPhysicalName(dim, grpTag, 'dom')
#
# #################################
# #        Write Mesh File        #
# #################################
# gmsh.write(meshPath)
# # if '-nopopup' not in sys.argv:
# gmsh.fltk.run()
# gmsh.finalize()

#
# projName = '2D_cylinder'
# # dom_dx, dom_dy = 60, 25
# centX, centY, centZ = 0, 0, 0
# cylD = 1
# domD = cylD * 20
# meshSizeMax = 0.5
# #################################
# #          Initialize           #
# #################################
# gmsh.initialize()
# # By default Gmsh will not print out any messages: in order to output messages
# # on the terminal, just set the "General.Terminal" option to 1:
# gmsh.option.setNumber("General.Terminal", 0)
# gmsh.clear()
# gmsh.model.add(projName)
# gmsh.option.setNumber('Mesh.MeshSizeFactor', meshSF)
# #################################
# #      YALES2 Requirements      #
# #################################
# # Make sure "Recombine all triangular meshes" is unchecked so only triangular elements are produced
# gmsh.option.setNumber('Mesh.RecombineAll', 0)
# # Only save entities that are assigned to a physical group
# gmsh.option.setNumber('Mesh.SaveAll', 0)
# #################################
# #           Geometry            #
# #################################
# cyl = gmsh.model.occ.addCircle(centX, centY, centZ, cylD)
# topPt = gmsh.model.occ.addPoint(centX, centY + domD, centZ)
# botPt = gmsh.model.occ.addPoint(centX, centY - domD, centZ)
# centPt = gmsh.model.occ.addPoint(centX, centY, centZ)
# inlet = gmsh.model.occ.addCircleArc(botPt, centPt, topPt)
# outlet = gmsh.model.occ.addCircleArc(topPt, centPt, botPt)
# outerLoop = gmsh.model.occ.addCurveLoop([inlet, outlet])
# innerLoop = gmsh.model.occ.addCurveLoop([cyl])
# dom = gmsh.model.occ.addPlaneSurface([innerLoop, outerLoop])
# # We finish by synchronizing the data from OpenCASCADE CAD kernel with
# # the Gmsh model:
# gmsh.model.occ.synchronize()
# #################################
# #    Physical Group Naming      #
# #################################
# dim = 2
# grpTag = gmsh.model.addPhysicalGroup(dim, [dom])
# gmsh.model.setPhysicalName(dim, grpTag, 'dom')
# dim = 1
# grpTag = gmsh.model.addPhysicalGroup(dim, [inlet])
# gmsh.model.setPhysicalName(dim, grpTag, 'x0')
# grpTag = gmsh.model.addPhysicalGroup(dim, [outlet])
# gmsh.model.setPhysicalName(dim, grpTag, 'x1')
# grpTag = gmsh.model.addPhysicalGroup(dim, [cyl])
# gmsh.model.setPhysicalName(dim, grpTag, 'cyl')
# #################################
# #           MESHING             #
# #################################
# # We could also use a `Box' field to impose a step change in element
# # sizes inside a box
# # boxF = gmsh.model.mesh.field.add("Box")
# # gmsh.model.mesh.field.setNumber(boxF, "VIn", meshSizeMax/10)
# # gmsh.model.mesh.field.setNumber(boxF, "VOut", meshSizeMax)
# # gmsh.model.mesh.field.setNumber(boxF, "XMin", cylD/3)
# # gmsh.model.mesh.field.setNumber(boxF, "XMax", cylD/3+cylD*10)
# # gmsh.model.mesh.field.setNumber(boxF, "YMin", -cylD)
# # gmsh.model.mesh.field.setNumber(boxF, "YMax", cylD)
# # # Finally, let's use the minimum of all the fields as the background mesh field:
# # minF = gmsh.model.mesh.field.add("Min")
# # gmsh.model.mesh.field.setNumbers(minF, "FieldsList", [boxF])
# # gmsh.model.mesh.field.setAsBackgroundMesh(minF)
#
# # Set minimum and maximum mesh size
# #gmsh.option.setNumber('Mesh.MeshSizeMin', meshSizeMin)
# gmsh.option.setNumber('Mesh.MeshSizeMax', meshSizeMax)
#
# # Set number of nodes along cylinder wall
# gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 200)
# # gmsh.option.setNumber('Mesh.MeshSizeFromCurvatureIsotropic', 1)
#
# # Set size of mesh at every point in model
# # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshSize)
#
# # gmsh.model.mesh.setTransfiniteCurve(cylCir, 150, coef=1.1)
# # We can then generate a 2D mesh...
# gmsh.model.mesh.generate(1)
# gmsh.model.mesh.generate(2)
# # extract number of elements
# # get all elementary entities in the model
# entities = gmsh.model.getEntities()
# e = entities[-1]
# # get the mesh elements for each elementary entity
# elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(
#     e[0], e[1])
# # count number of elements
# numElem = sum(len(i) for i in elemTags)
# # ... and save it to disk
# gmsh.write(meshPath)
# # To visualize the model we can run the graphical user interface with
# # `gmsh.fltk.run()'.
# gmsh.fltk.run()
# # This should be called when you are done using the Gmsh Python API:
# gmsh.finalize()

# import os
# import subprocess
# # caseDir = os.path.join(os.getcwd(), 'test_case')
# # caseDir = 'test_case'
# # solverExecCmd = ['cd', 'test_case', '&&', 'mpirun', '-n',
# #                  '10', '2D_cylinder', '>', 'pyTest.out']
# # print(solverExecCmd)
# # print(caseDir)
# subprocess.run(['mpirun', '-n', '10', '2D_cylinder'],
#                cwd='test_case', stdout=subprocess.DEVNULL)

# import numpy as np
#
# A = np.around(np.arange(0.5, 1.5, 0.1), decimals=2)
# print(A)
# print([a for a in A])
# A = np.around(A, decimals=2)
# print(A)
# print([a for a in A])


# numpy.around(a, decimals=0)


# lines = ['asfasdf/n', 'assdasdf/n', 'sfassdf/n', 'sdfsf']
#
# for i, line in enumerate(lines):
#     if not line.endswith('\n'):
#         lines[i] += '\n'
#
# print(lines)


# import numpy as np
# print(int(23/5))

# class A:
#     @classmethod
#     def a(cls):
#         print('A')
#
# class B(A):
#     @classmethod
#     def a(cls):
#         print('B')
#
#     @staticmethod
#     def b(pts):
#         for pt in pts:
#
# b = B()
# b.a()
# B.a()
# A.a()
# A().a()

# import os
# import numpy as np
# print('scratch.py')
#
# from pymooCFD.studies.oscillCyl import BaseCase, MyOptStudy
#
# wds = ['test_case1', 'test_case2', 'test_case3']
# X = [[1.5, 0.1], [0.5, 0.2], [1, 0.3]]
# cases = []
# for i, wd in enumerate(wds):
#     cases.append(BaseCase('osc-cyl_base', wd, X[i]))
#
# from setupOpt import algorithm, problem
# MyOptStudy(algorithm, problem, BaseCase).runPop(cases)

# BaseCase('base_cases/osc-cyl_base', 'test_case', [0,0]).genMesh()

# import subprocess
# import multiprocessing as mp
# from tqdm import tqdm
#
# NUMBER_OF_TASKS = len(wds)
#
# procLim = 40
# nProc = 20
# nTask = int(procLim/nProc)
# progress_bar = tqdm(total=nTask)
#
#
# def solve(wd):
#     # command = ['python', 'worker.py', sec_sleep]
#     cmd = ['mpirun', '-n', str(nProc), '2D_cylinder']
#     print(cmd)
#     print(wd)
#     subprocess.run(cmd, cwd=wd, stdout=subprocess.DEVNULL)
#     print('COMPLETE', wd)
#
#
# def execCallback():
#     if not _execDone():
#         pool.apply_async(solve, (wd,), callback=execCallback)
#
#
# def _execDone():
#     return True
#     # progress_bar.update()
#
#
# if __name__ == '__main__':
#
#     pool = mp.Pool(nTask)
#
#     # for seconds in [str(x) for x in range(1, NUMBER_OF_TASKS + 1)]:
#     for wd in wds:
#         pool.apply_async(solve, (wd,), callback=execCallback)
#
#     # check if simulation completed correctly
#     # if not _execDone():
#     #     pass
#
#     pool.close()
#     pool.join()


# import subprocess
# cmd = ['mpirun', '2D_cylinder']
# wds = ['test_case1', 'test_case2', 'test_case3']
#
# for wd in wds:
#     subprocess.Popen(cmd, cwd=wd)

# # def loadCases(directory):
# directory = os.path.join('opt_run', 'gen1')
# print(os.listdir(directory))
# cases = []
# ents = os.listdir(directory)
# for ent in ents:
#     ent_path = os.path.join(directory, ent)
#     if os.path.isdir(ent_path):
#         for e in os.listdir(ent_path):
#             caseCP = os.path.join(ent_path, 'case.npy')
#             if os.path.exists(caseCP):
#                 case, = np.load(caseCP, allow_pickle=True).flatten()
#                 cases.append(case)
# # return cases
# # loadCases('opt_run/gen1')


# import numpy as np
# class Test:
#     var = 1
#     def func(self):
#         print(self.var)
#     def __str__(self):
#         return 'TEST'
#
# test1 = Test()
# test2 = Test()
# l = np.array([test1, test2])
# np.save('cpTest.npy', l, allow_pickle=True)
# print(l)
# list = np.load('cpTest.npy', allow_pickle=True).flatten()
# print(list)


# import numpy as np
# class Test:
#     def __init__(self):
#         self.l = np.array([0.006599578695870828, 0.35595787774394816])
#     def __str__(self):
#         return f'{self.l}'
# # l = [0.006599578695870828, 0.35595787774394816]
# test = Test()
# print(test.l[0])
# # print(test.l)
# # print(f'{test.l}')
# print(test)
# print(f'{test}')
# print(type(test.l))
# # with np.printoptions(suppress=True):
# #     print(test.l)


# class Test:
#     var = 1
#     def func(self):
#         print(self.var)
#     def __str__(self):
#         return 'TEST'
#
# test1 = Test()
# test2 = Test()
# test1.__str__ = lambda self: 'Test 1'
# l = [[1, test1], [2, test2]]
# for e in l: print(f'\t\t {e[0]} {e[1]}')


# # from http://stackoverflow.com/questions/4103773/efficient-way-of-having-a-function-only-execute-once-in-a-loop
# import functools
# def run_once(f):
#     """Runs a function (successfully) only once.
#
#     The running can be reset by setting the `has_run` attribute to False
#     """
#     @functools.wraps(f)
#     def wrapper(*args, **kwargs):
#         if not wrapper.complete:
#             result = f(*args, **kwargs)
#             wrapper.complete = True
#             return result
#     wrapper.complete = False
#     return wrapper
#
# def calltracker(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         result = func(*args, **kwargs)
#         wrapper.complete = True
#         return result
#     wrapper.complete = False
#     return wrapper
#
# @run_once
# def test1():
#     print('test1')
#
# @calltracker
# def test2():
#     print('test2')
#     print(test2.complete)
#
# test1()
# test1()
#
# test2()
# test2()
#
# import numpy
# numpy.save('test2', test2)
# test2, = numpy.load('test2.npy', allow_pickle=True).flatten()
# test2()
#
# numpy.save('test1', test1)
# test1, = numpy.load('test1.npy', allow_pickle=True).flatten()
# test1()

# import shutil
# shutil.copytree('base_case', 'test_case')

# xl = [1, 2, 5]
# xu = [3, 4, 6]
# n_var= 3
# x_mid = [xl[x_i]+(xu[x_i]-xl[x_i])/2 for x_i in range(n_var)]
# print(x_mid)
