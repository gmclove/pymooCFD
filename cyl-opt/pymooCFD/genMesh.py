import gmsh
import sys
import os
import math

import numpy as np


meshSizeMin =  1e-6
meshSizeMax = 0.01


def main():
    genMesh('jet_cone-axi_sym.unv', 1, 0.02)
    # step = 0.05
    # # meshSF = step:step:0.3
    # meshSF = np.arange(step, 0.3, step)
    # print(meshSF)
    # meshSFs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # for meshSF in meshSFs:
    #     genMesh('.', meshSF, [0.02])

    # genMesh('.', 1, [0.02])


def genMesh(meshPath, meshSF, outD):
    projName = 'jet_cone_axi-sym'

    cyl_r = outD/2 #0.02/2

    sph_r = (0.15/0.02)*cyl_r
    sph_xc = -sph_r
    sph_yc = 0

    cone_dx = 3.5
    cone_r1 = 0.5/2
    cone_r2 = cone_dx*(0.7/2.5)+cone_r1
    cone_xc = -0.5

    #################################
    #          Initialize           #
    #################################
    gmsh.initialize(sys.argv)
    gmsh.model.add(projName)
    gmsh.option.setNumber('General.Terminal', 0)
    #################################
    #      YALES2 Requirements      #
    #################################
    # Make sure "Recombine all triangular meshes" is unchecked so only triangular elements are produced
    gmsh.option.setNumber('Mesh.RecombineAll', 0)
    # Only save entities that are assigned to a physical group
    gmsh.option.setNumber('Mesh.SaveAll', 0)

    #################################
    #           Geometry            #
    #################################
    ##### Points #####
    # Cone
    cone_ll = gmsh.model.occ.addPoint(cone_xc, 0, 0)
    cone_lr = gmsh.model.occ.addPoint(cone_dx+cone_xc, 0, 0)
    cone_ur = gmsh.model.occ.addPoint(cone_dx+cone_xc, cone_r2, 0)
    cone_ul = gmsh.model.occ.addPoint(cone_xc, cone_r1, 0)
    # sphere
    sph_start = gmsh.model.occ.addPoint(sph_xc-sph_r, sph_yc, 0)
    sph_cent = gmsh.model.occ.addPoint(sph_xc, sph_yc, 0)
    sph_end_x = sph_xc + math.sqrt(sph_r**2 - cyl_r**2)
    sph_end = gmsh.model.occ.addPoint(sph_end_x, sph_yc+cyl_r, 0)
    # cylinder
    cyl_ll = sph_cent
    cyl_ul = gmsh.model.occ.addPoint(sph_xc, sph_yc+cyl_r, 0)
    cyl_ur = sph_end
    cyl_mid_outlet = gmsh.model.occ.addPoint(sph_xc+sph_r, 0, 0)


    ##### Curves #####
    # sphere wall
    sph_wall = gmsh.model.occ.addCircleArc(sph_start, sph_cent, sph_end)
    # cylinder wall
    cyl_wall = gmsh.model.occ.addLine(cyl_ul, cyl_ur)
    # cylinder axis (for meshing control)
    axis_cyl = cyl_axis = gmsh.model.occ.addLine(cyl_ll, cyl_mid_outlet)
    # inlet (cylinder back wall)
    inlet = gmsh.model.occ.addLine(cyl_ul, cyl_ll)
    # axis
    axis_l = gmsh.model.occ.addLine(sph_start, cone_ll)
    axis_r = gmsh.model.occ.addLine(cyl_mid_outlet, cone_lr)
    axis = [axis_l, axis_cyl, axis_r]
    # cone
    back_wall = gmsh.model.occ.addLine(cone_ll, cone_ul)
    cone_wall = gmsh.model.occ.addLine(cone_ul, cone_ur)
    outlet = gmsh.model.occ.addLine(cone_ur, cone_lr)
    # mesh field line
    # field_line = gmsh.model.occ.addLine(sph_end, cone_ur)

    ##### Surfaces #####
    # domain surface
    curv_loop_tags = [sph_wall, cyl_wall, inlet, axis_cyl, axis_r, outlet,
                      cone_wall, back_wall, axis_l]
    dom_loop = gmsh.model.occ.addCurveLoop(curv_loop_tags)
    dom = gmsh.model.occ.addPlaneSurface([dom_loop])

    ##### Mesh Construction Surface(s) #####
    # mesh construction line
    # mesh_pt1 = gmsh.model.occ.addPoint(sph_r, cyl_r + 0.01, 0)
    # mesh_pt2 = gmsh.model.occ.addPoint(cone_dx + cone_xc, cone_r2, 0)
    # mesh_line = gmsh.model.occ.addLine(mesh_pt1, mesh_pt2)



    # mesh field surface
    # field_loop = gmsh.model.occ.addCurveLoop([cyl_wall, cyl_r, axis_r, outlet,
    #                                           field_line])
    # field_surv = gmsh.model.occ.addPlaneSurface([field_loop])

    # sph = gmsh.model.occ.addSphere(sph_xc, sph_yc, sph_zc, sph_r)
    # cyl = gmsh.model.occ.addCylinder(sph_xc, sph_yc, sph_zc, sph_r, 0, 0, cyl_r)
    # gmsh.model.occ.rotate([(3,cyl)], sph_xc, sph_yc, sph_zc, 0, 0, 1, out_angle)
    # cutDimTags, cutDimTagsMap = gmsh.model.occ.cut([(3, sph)], [(3, cyl)] ) #, removeObject=False, removeTool=False)

    # dom = gmsh.model.occ.addCone(cone_xc, cone_yc, cone_zc, cone_dx, cone_dy, cone_dz, cone_r1, cone_r2)
    # domDimTags, domDimTagsMap = gmsh.model.occ.cut([(3, dom)], cutDimTags) #, removeTool=False)


    # box = gmsh.model.occ.addBox(sph_xc-sph_r, sph_yc-sph_r, sph_zc-sph_r, 2*sph_r, 2*sph_r-out_h, 2*sph_r)
    # gmsh.model.occ.rotate([(3,box)], sph_xc, sph_yc, sph_zc, 0, 0, 1, out_angle + math.pi/2)
    # capDimTags, capDimTagsMap = gmsh.model.occ.cut([(3, sph)], [(3, box)], removeObject=False)
    # capDimTags, capDimTagsMap = gmsh.model.occ.fragment([(3, sph)], [(3, box)], removeTool=True, removeObject=True)

    # gmsh.model.occ.remove(cutDimTags, recursive=True)

    ### REMOVAL FOR VISUALIZATION BUT ONLY ENTITIES ASSIGNED TO PHYSICAL GROUPS ARE WRITTEN TO MESH FILE
    # remove whole sphere surface
    # gmsh.model.occ.remove([(2, 7)])
    # # remove bottom of spherical cap
    # gmsh.model.occ.remove([(2, 15)])
    # gmsh.model.occ.remove(capDimTags)
    # # remove box and all its entities recursively
    # for i in range(6):
    #     gmsh.model.occ.remove([(2, 17 + i)], recursive=True)

    # in_tag = 4
    # sph_wall = 5
    # cyl_wall = 6
    # cone_walls = [7, 9]
    # head_tag = 14
    # walls_start_tag = 86
    gmsh.model.occ.synchronize()
    #################################
    #           MESHING             #
    #################################
    ##### Field Meshing ######
    ### Distance and Threshold Field
    # dist_fld = gmsh.model.mesh.field.add('Distance')
    # gmsh.model.mesh.field.setNumbers(1, "PointsList", [cyl_mid_outlet])
    # gmsh.model.mesh.field.setNumbers(dist_fld, "CurvesList", [axis_r])
    # gmsh.model.mesh.field.setNumber(dist_fld, "NumPointsPerCurve", 700)
    # SizeMax -                     /------------------
    #                              /
    #                             /
    #                            /
    # SizeMin -o----------------/
    #          |                |    |
    #        Point         DistMin  DistMax
    # thres_fld = gmsh.model.mesh.field.add("Threshold")
    # gmsh.model.mesh.field.setNumber(thres_fld, "InField", 1)
    # gmsh.model.mesh.field.setNumber(thres_fld, "SizeMin", 0.002)
    # gmsh.model.mesh.field.setNumber(thres_fld, "SizeMax", 0.03)
    # gmsh.model.mesh.field.setNumber(thres_fld, "DistMin", cyl_r*2)
    # gmsh.model.mesh.field.setNumber(thres_fld, "DistMax", 0.5)
    # gmsh.model.mesh.field.setNumber(thres_fld, "Sigmoid", 0)



    ### Clyinder field
    # cyl_fld_dx = 3
    # rise = cone_r2 - cone_r1
    # run = 3
    # slope = rise/run
    # yaxis = slope*cyl_fld_dx
    # # xaxis = (cone_r1 / cone_r2) *
    # cyl_fld_r = 0.5
    # cyl_field = gmsh.model.mesh.field.add('Cylinder')
    # gmsh.model.mesh.field.setNumber(cyl_field, 'Radius', cyl_fld_r)
    # gmsh.model.mesh.field.setNumber(cyl_field, 'VIn', 0.005)
    # gmsh.model.mesh.field.setNumber(cyl_field, 'VOut', meshSizeMax)
    # gmsh.model.mesh.field.setNumber(cyl_field, 'XCenter', 1)
    # gmsh.model.mesh.field.setNumber(cyl_field, 'YCenter', -cyl_fld_r/2)
    # gmsh.model.mesh.field.setNumber(cyl_field, 'ZCenter', 0)
    # gmsh.model.mesh.field.setNumber(cyl_field, 'XAxis', cyl_fld_dx)
    # gmsh.model.mesh.field.setNumber(cyl_field, 'YAxis', yaxis)
    # gmsh.model.mesh.field.setNumber(cyl_field, 'ZAxis', 0)
    # ### Smaller Cylinder
    # cyl_fld2_dx = 0.5
    # cyl_fld2_dy = slope*cyl_fld2_dx
    # cyl_fld2 = gmsh.model.mesh.field.add('Cylinder')
    # gmsh.model.mesh.field.setNumber(cyl_fld2, 'Radius', cyl_fld_r)
    # gmsh.model.mesh.field.setNumber(cyl_fld2, 'VIn', 0.003)
    # gmsh.model.mesh.field.setNumber(cyl_fld2, 'VOut', meshSizeMax)
    # gmsh.model.mesh.field.setNumber(cyl_fld2, 'XCenter', 1/2)
    # gmsh.model.mesh.field.setNumber(cyl_fld2, 'YCenter', -0.35)
    # gmsh.model.mesh.field.setNumber(cyl_fld2, 'ZCenter', 0)
    # gmsh.model.mesh.field.setNumber(cyl_fld2, 'XAxis', cyl_fld2_dx)
    # gmsh.model.mesh.field.setNumber(cyl_fld2, 'YAxis', cyl_fld2_dy)
    # gmsh.model.mesh.field.setNumber(cyl_fld2, 'ZAxis', 0)



    ### typically finalize multiple field meshing by setting minumim value of
    ### fields in use as the background mesh
    # min_field = gmsh.model.mesh.field.add('Min')
    # gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [thres_fld]) #cyl_field, cyl_fld2])
    # gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    ##### Transfinite Meshing #####
    # gmsh.model.mesh.setSize([(0, cyl_ll)], 0.001)

    gmsh.model.mesh.setTransfiniteCurve(sph_wall, 100)
    NN_cyl_axis = 70
    gmsh.model.mesh.setTransfiniteCurve(cyl_wall, NN_cyl_axis)
    gmsh.model.mesh.setTransfiniteCurve(cyl_axis, NN_cyl_axis)
    gmsh.model.mesh.setTransfiniteCurve(inlet, 20)
    gmsh.model.mesh.setTransfiniteCurve(axis_r, 500, meshType='Progression', coef=1.005)
    gmsh.model.mesh.setTransfiniteCurve(axis_l, 40)
    gmsh.model.mesh.setTransfiniteCurve(back_wall, 15)
    gmsh.model.mesh.setTransfiniteCurve(outlet, 50, meshType='Progression', coef=0.99)
    gmsh.model.mesh.setTransfiniteCurve(cone_wall, 100)

    ##### General Meshing Criteria ######
    # gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    # gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 30)

    # gmsh.model.mesh.setSizeFromBoundary(2, 8, 1)

    # Set minimum and maximum mesh size
    # gmsh.option.setNumber('Mesh.MeshSizeMin', meshSizeMin)
    # gmsh.option.setNumber('Mesh.MeshSizeMax', meshSizeMax)


    # gmsh.option.setNumber('Mesh.MeshSizeFactor', meshSF)

    ##### Execute Meshing ######
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.generate(2)
    # gmsh.model.mesh.generate(3)

    # print(cyl_axis)
    # print(sph)
    # print(cyl)
    # print(cutDimTags)
    # print(cutDimTagsMap)
    # print(domDimTags, domDimTagsMap)
    # print(slope)
    #################################
    #    Physical Group Naming      #
    #################################
    # 1-D physical groups
    dim = 1
    # inlet
    grpTag = gmsh.model.addPhysicalGroup(dim, [inlet])
    gmsh.model.setPhysicalName(dim, grpTag, 'inlet')
    # outlet
    grpTag = gmsh.model.addPhysicalGroup(dim, [outlet])
    gmsh.model.setPhysicalName(dim, grpTag, 'outlet')
    # axis of symmetry
    grpTag = gmsh.model.addPhysicalGroup(dim, axis)
    gmsh.model.setPhysicalName(dim, grpTag, 'axis')
    # walls
    grpTag = gmsh.model.addPhysicalGroup(dim, [sph_wall, cyl_wall])
    gmsh.model.setPhysicalName(dim, grpTag, 'walls')
    # coflow
    grpTag = gmsh.model.addPhysicalGroup(dim, [back_wall, cone_wall])
    gmsh.model.setPhysicalName(dim, grpTag, 'coflow')

    # 2-D physical groups
    dim = 2
    grpTag = gmsh.model.addPhysicalGroup(dim, [dom])
    gmsh.model.setPhysicalName(dim, grpTag, 'dom')


    #################################
    #        Write Mesh File        #
    #################################
    gmsh.write(meshPath)
    # gmsh.write(os.path.join(caseDir, projName + '.unv'))


    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()
    gmsh.finalize()

#if __name__ == '__main__':
#    main()
