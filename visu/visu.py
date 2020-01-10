# Visualization of 3D point clouds (like graph visualisation)

import open3d
import torch
from visu.graph_manipulation import list_neigh2list_edge



def visu_pcd(data, scales=[0], is_graph=True, color=None):
    """
    visualize a point cloud using open3d
    inputs:
    data (MultiScaleData): contain the list of points cloud in data.points (in torch format)
    is_multi_scale (bool): display all the scale if True (if False display only the first scale)
    is_graph (bool): display the links between graphs
    color: color of the point cloud
    """

    for i in scales:
        pt = data.points[i]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pt.numpy())
        if(color is not None):
            pcd.paint_uniform_color(color)
        if(is_graph):
            line_set = open3d.geometry.LineSet()

            edges = list_neigh2list_edge(data.list_neigh[i]).tolist()

            line_set.points = open3d.utility.Vector3dVector(pt.tolist())
            line_set.lines = open3d.utility.Vector2iVector(edges)
            open3d.visualization.draw_geometries([pcd, line_set])
        else:
            open3d.visualization.draw_geometries([pcd])
