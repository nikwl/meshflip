import os
import json

import vtk
import vedo
import trimesh
import numpy as np
import open3d as o3d
import pyransac3d as pyrsc
from sklearn.decomposition import PCA


# === conversion methods === 

def trimesh2vedo(mesh, **kwargs):
    """ Convert a trimesh mesh to a vedo mesh """
    try:
        return vedo.Mesh(
            [mesh.vertices, mesh.faces], 
            **kwargs
        )
    except AttributeError:
        return vedo.Mesh(
            [mesh.vertices, None], 
            **kwargs
        )


def vedo2trimesh(mesh):
    """ Convert a vedo mesh to a trimesh mesh """
    try:
        return trimesh.Trimesh(
            vertices=mesh.points(), 
            faces=mesh.faces(),
        )
    except AttributeError: 
        return trimesh.Trimesh(
            vertices=mesh.points(),
        )


def o3d2trimesh(pc):
    """ Convert an open3d pointcloud to a trimesh mesh """
    points = pc.points
    colors = pc.colors
    pc = trimesh.Trimesh(points)
    if colors is not None:
        pc.visual = trimesh.visual.color.ColorVisuals(
            mesh=pc,
            vertex_colors=np.ones((pc.vertices.shape[0], 4)).astype(np.uint8) * 255
        )
    return pc


def force_trimesh(mesh, remove_texture=False):
    """ Take a trimesh mesh, scene, pointcloud, or list of these and force it to be a trimesh mesh """

    if isinstance(mesh, list):
        return [force_trimesh(m) for m in mesh]
    
    if isinstance(mesh, trimesh.PointCloud):
        return trimesh.Trimesh(mesh.vertices, vertex_colors=mesh.colors)
    elif isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            mesh = trimesh.Trimesh()
        else:
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in mesh.geometry.values()))
    if remove_texture:
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    return mesh

# === trimesh methods === 

def trimesh_rotate(mesh, rotx=0, roty=0, rotz=0):
    """ Apply a sequence of rotations to a trimesh mesh """
    mesh = trimesh2vedo(mesh)
    transform = vtk.vtkTransform()
    transform.RotateX(rotx)
    transform.RotateY(roty)
    transform.RotateZ(rotz)
    mesh.SetUserTransform(transform)
    return vedo2trimesh(mesh)


def trimesh_pointcloud_concat(pointclouds):
    """ Concatonate a list of trimesh pointclouds """
    point_cloud_accumulator = trimesh.util.concatenate(
        force_trimesh(pointclouds)
    )
    return trimesh.PointCloud(
        vertices=point_cloud_accumulator.vertices,
        colors=point_cloud_accumulator.visual.vertex_colors
    )


def trimesh_bbx(mesh):
    """ Get the bounding box of a trimesh mesh or pointcloud """
    return (
        mesh.vertices.min(axis=0),
        mesh.vertices.max(axis=0),
    )


def trimesh_center(mesh):
    """ Translate a trimesh mesh or pointcloud to its centroid """
    mesh = mesh.copy()
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    return mesh


def trimesh_transform_matrix(mesh, mat):
    """ Apply a transformation matrix to a trimesh mesh or pointcloud """
    mesh = mesh.copy()
    mesh.apply_transform(mat)
    return mesh


def trimesh_transform(mesh, R, T, inverse=False):
    """ Apply a rotation and translation to a trimesh mesh or pointcloud """
    mesh = mesh.copy()
    if inverse:
        mesh.vertices = (np.dot(R, (mesh.vertices - T.flatten()).T)).T.astype(np.float32)
    else:
        mesh.vertices = (np.dot(R.T, mesh.vertices.T).T + T.flatten()).astype(np.float32)
    return mesh


def trimesh_normalize(mesh):
    """Normalize a mesh so that it occupies a unit cube """

    # Get the overall size of the object
    mesh = mesh.copy()
    mesh_min, mesh_max = np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)
    size = mesh_max - mesh_min

    # Center the object
    mesh.vertices = mesh.vertices - ((size / 2.0) + mesh_min)

    # Normalize scale of the object
    mesh.vertices = mesh.vertices * (1.0 / np.max(size))
    return mesh


def trimesh_normalize_matrix(mesh):
    """Normalize a mesh so that it occupies a unit cube """

    # Get the overall size of the object
    mesh_min, mesh_max = np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)
    size = mesh_max - mesh_min

    # Center the object
    mesh.vertices = ((size / 2.0) + mesh_min)

    trans = trimesh.transformations.translation_matrix(
        -((size / 2.0) + mesh_min)
    )

    # Normalize scale of the object
    scale = trimesh.transformations.scale_matrix(
        (1.0 / np.max(size))
    )
    return scale @ trans


def trimesh_settle(mesh, search_height=None, component_points=10, ransac_threshold=0.01, return_matrix=False):
    mesh = mesh.copy()

    # Get the mesh boundingbox
    search_bbx = trimesh_bbx(mesh)

    # Get a point corresponding to the bottom of the bounding box + search_height
    if search_height is None:
        search_height = (search_bbx[1][2] - search_bbx[0][2]) * 0.1
    max_point = search_bbx[1]
    max_point[2] += search_height

    # Slice the mesh along that plane
    mesh = trimesh.intersections.slice_mesh_plane(mesh, (0, 0, -1), max_point)

    # Split into connected components
    components = [vedo2trimesh(m) for m in trimesh2vedo(mesh).splitByConnectivity()]

    # For each component, take the lowest x points
    points = []
    for c in components:
        points.append(
            c.vertices[np.argsort(c.vertices[:, 2])[:component_points], :]
        )
    points = np.vstack(points)

    # Find a plane that passes through these points
    _, n = points_find_plane(
        points, 
        ransac_threshold=ransac_threshold
    )

    # The normal may be in the wrong direction, take the one that's closest to the up vector
    angle1 = trimesh.transformations.angle_between_vectors(n, (0, 0, 1))
    angle2 = trimesh.transformations.angle_between_vectors(-n, (0, 0, 1))
    if angle2 < angle1:
        n = -n

    # Compute the matrix
    mat = trimesh.transformations.rotation_matrix(
        trimesh.transformations.angle_between_vectors(n, (0, 0, 1)), 
        trimesh.transformations.vector_product(n, (0, 0, 1)),
    )

    mesh = trimesh_transform_matrix(mesh, mat)
    if return_matrix:
        return mesh, mat
    return mesh


# === point set methods ===


def points_transform(vs, R, T, inverse=False):
    """ Apply a rotation and translation to a set of points """
    if inverse:
        return (np.dot(R, (vs - T.flatten()).T)).T
    return (np.dot(R.T, vs.T).T + T.flatten())


def points_transform_matrix(vs, mat):
    """ Apply a transformation matrix to a set of points """
    return np.dot(
        mat, 
        np.hstack((
            vs, 
            np.ones((vs.shape[0], 1))
        )).T
    ).T[:, :3]


def points_crop_3d(vs, bbx, return_mask=False):
    """ Crop a set of points using a 3d bounding box """
    mask = np.vstack((
        vs[:, 0] > bbx[0][0], 
        vs[:, 0] < bbx[1][0],
        vs[:, 1] > bbx[0][1], 
        vs[:, 1] < bbx[1][1],
        vs[:, 2] > bbx[0][2], 
        vs[:, 2] < bbx[1][2],
    )).all(axis=0)

    if return_mask:
        return vs[mask, :], mask
    return vs[mask, :]


def points_find_plane(points, ransac_threshold=0.01):
    """ Find the largest plane from a set of points """
    
    # Fit the plane
    best_eq, _ = pyrsc.Plane().fit(points, ransac_threshold)

    # Convert to point normal form
    n = best_eq[:3]
    n = n / np.dot(n, n)
    p = [0, 0, -float(best_eq[3] / best_eq[2])]

    return p, n


def points_remove_plane(vs, p, n, return_mask=False):
    """ Remove points from a set of points that are below a plane """
    mask = np.dot(vs - p, n) > 0
    if return_mask:
        return vs[mask, :], mask
    else:
        return vs[mask, :]


def points_find_remove_plane(
    points, 
    ransac_threshold=0.01,
    plane_threshold=40, # How far to move the plane before removing points 
    return_plane=False
):  
    """ Find and remove the largest plane from a set of points """

    # Get num points
    orig_num_pts = points.shape[0]
    
    # Fit the plane
    p, n = points_find_plane(points, ransac_threshold)

    # Do initial slice
    sliced_vertices = points_remove_plane(points, p, n)

    # Re-orient normal vector
    if sliced_vertices.shape[0] < (orig_num_pts * 0.5):
        n = -n

    # Edge plane up a little and cut
    p_cur = p + (n * plane_threshold)
    sliced_vertices, mask = points_remove_plane(points, p_cur, n, return_mask=True)

    if return_plane:
        return sliced_vertices, mask, n, p_cur
    return sliced_vertices, mask


def points_maximal_orient(points):
    """ Return the transformation matrix that orients a point set by its maximal dimensions """
    pca = PCA(n_components=3)
    pca.fit(points)
    matrix = pca.components_
    return np.vstack((
        np.hstack((
            np.expand_dims(matrix[2, :], axis=1),
            np.expand_dims(matrix[1, :], axis=1),
            np.expand_dims(matrix[0, :], axis=1),
            np.zeros((3, 1))
        )),
        np.array([0, 0, 0, 1])
    )).T


def points_icp(moving, fixed, threshold=75, scale=False):
    """ Align two point sets using icp """
    pc_moving = o3d.geometry.PointCloud()
    pc_moving.points = o3d.utility.Vector3dVector(moving)
    pc_fixed = o3d.geometry.PointCloud()
    pc_fixed.points = o3d.utility.Vector3dVector(fixed)
    return o3d.pipelines.registration.registration_icp(
        pc_moving, 
        pc_fixed,
        threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=scale),
    ).transformation

# === transformation methods ===


def is_transformation_matrix(mat):
    """ Rough double check that this is a transformation matrix """
    assert np.isclose(np.linalg.det(mat[:3, :3]), 1, atol=1e-4), "Bad rotation matrix"
    assert (mat[3, :] == np.array([0, 0, 0, 1])).all(), "Bad matrix"


def create_transformation_matrix(R, T):
    """ Create a transformation matrix from rotation and translation components """
    return np.hstack(
        (np.vstack([R,T]), np.expand_dims(np.array([0, 0, 0, 1]), axis=1))
    )


def load_transform(path, return_data=False):
    """ load transform from a .json file """
    assert os.path.splitext(path)[-1] == ".json", "file must be a .json"
    data = json.load(open(path))
    data["matrix"] = np.asarray(data["matrix"])
    if return_data:
        return data
    return data["matrix"]


def save_transform(path, matrix, **kwargs):
    """ save transform to a .json file """
    assert os.path.splitext(path)[-1] == ".json", "file must be a .json"
    kwargs["matrix"] = matrix.tolist()
    json.dump(kwargs, open(path, "w"))


def load_transform_list(path, return_data=False):
    """ Load a list of transforms from a .json file """
    assert os.path.splitext(path)[-1] == ".json", "file must be a .json"
    data = json.load(open(path))
    assert isinstance(data["transforms"], list)
    data["transforms"] = [np.array(tf) for tf in data["transforms"]]
    if return_data:
        return data
    return data["transforms"]


def save_transform_list(path, transforms, **kwargs):
    """ Save a list of transforms to a .json file """
    assert os.path.splitext(path)[-1] == ".json", "file must be a .json"
    assert isinstance(transforms, list)
    kwargs["transforms"] = [tf.tolist() for tf in transforms]
    json.dump(kwargs, open(path, "w"))


def load_plane(path, return_data=False):
    """ Load a plane from a .json file """
    assert os.path.splitext(path)[-1] == ".json", "file must be a .json"
    data = json.load(open(path))
    data["point"] = np.array(data["point"])
    data["normal"] = np.array(data["normal"])
    if return_data:
        return data
    return data["point"], data["normal"]


def save_plane(path, point, normal, **kwargs):
    """ Save a plane to a .json file """
    assert os.path.splitext(path)[-1] == ".json", "file must be a .json"
    kwargs["point"] = point.tolist()
    kwargs["normal"] = normal.tolist()
    json.dump(kwargs, open(path, "w"))


def load_bounding_box(path, return_data=False):
    """ Load a bounding box from a .json file """
    assert os.path.splitext(path)[-1] == ".json", "file must be a .json"
    data = json.load(open(path))
    data["bounding_box"] = np.array(data["bounding_box"])
    if return_data:
        return data
    return data["bounding_box"]


def save_bounding_box(path, bounding_box, **kwargs):
    """ Save a bounding box to a .json file """
    assert os.path.splitext(path)[-1] == ".json", "file must be a .json"
    try:
        kwargs["bounding_box"] = bounding_box.tolist()
    except AttributeError:
        kwargs["bounding_box"] = bounding_box
    json.dump(kwargs, open(path, "w"))
