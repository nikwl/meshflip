import argparse
import logging
import json

import trimesh
import vedo
import vtk
import numpy as np
from vedo import Plotter

from meshflip.logger import LOG
import meshflip.logger as logger
import meshflip.utils_3d as utils_3d


def get_mat(transform):
    """
    Get the matrix from a vtk transform object
    """
    return np.array(
        [transform.GetMatrix().GetElement(i, j) for i in range(4) for j in range(4)]
    ).reshape(4, 4)


def trimesh2vedo(mesh, **kwargs):
    """Convert a trimesh mesh to a vedo mesh"""
    try:
        return vedo.Mesh([mesh.vertices, mesh.faces], **kwargs)
    except AttributeError:
        return vedo.Mesh([mesh.vertices, None], **kwargs)


def register(
    moving, fixed=None, tf=None, output=None, icp_threshold=75, icp_scale=False
):

    # Instantiate the plotter
    plt = Plotter(axes=0)
    plt += """
    Instructions: Register the red object to the blue object.
    Click and drag to move the camera or the mesh. Use the middle mouse button to translate the mesh.
    Press "a" to toggle between moving the object or moving the camera.
    press "r" to register the objects using ICP.
    Press "s" to save the transformation to disk.
    Press "c" to correct the fixed mesh (if you moved it by accident).
    Press "q" to exit.
    """

    # Load the fixed and moving pointlclouds
    input_mesh = utils_3d.force_trimesh(moving)
    mesh_moving = trimesh2vedo(
        input_mesh,
        # c="r",
    )
    # if len(mesh_moving.faces()) == 0:
    #    mesh_moving = trimesh2vedo(mesh_moving.points(), c="r")
    plt.show(mesh_moving, interactive=False)

    if fixed is None:
        mesh_fixed = vedo.shapes.Plane(
            pos=(0, 0, 0), normal=(0, 0, 1), sx=5000, sy=None, c="b", alpha=0.2
        )
    else:
        mesh_fixed = trimesh2vedo(
            utils_3d.force_trimesh(fixed),
            # c="b",
        )
        # if len(mesh_fixed.faces()) == 0:
        #    mesh_fixed = vedo.pointcloud.Points(mesh_fixed.points(), c="b")
    plt.show(mesh_fixed, interactive=False)

    # Set a starting identity transform
    transform = vtk.vtkTransform()
    transform.SetMatrix(np.eye(4).flatten().flatten())
    mesh_moving.SetUserTransform(transform)

    mesh_moving._scale_matrix = np.eye(4)

    bbx_moving = utils_3d.trimesh_bbx(utils_3d.vedo2trimesh(mesh_moving))
    bbx_fixed = utils_3d.trimesh_bbx(utils_3d.vedo2trimesh(mesh_fixed))
    scale = (bbx_moving[0][0] - bbx_moving[1][0]) / (bbx_fixed[0][0] - bbx_fixed[1][0])

    def update_transform(mat):
        """Updates the user transform of the moving mesh"""
        mat = mat @ get_mat(mesh_moving.GetUserTransform())
        # assert np.isclose(np.linalg.det(mat[:3, :3]), 1, atol=1e-4)
        assert (mat[3, :] == np.array([0, 0, 0, 1])).all()
        transform = vtk.vtkTransform()
        transform.SetMatrix(mat.flatten())
        mesh_moving.SetUserTransform(transform)

    update_transform(np.eye(4))

    def scale(widget, event):
        scale = widget.GetRepresentation().GetValue()
        mesh_moving._scale_matrix = trimesh.transformations.scale_matrix(
            scale, [0, 0, 0]
        )
        update_transform(np.eye(4))

    def keypress_callback(evt):
        """
        Keypress callback, handle icp and applying optimal transform
        """
        if evt["keyPressed"] == "r":
            LOG.debug("Running ICP...")

            transform = vtk.vtkTransform()
            transform.SetMatrix(np.eye(4).flatten())
            mesh_fixed.SetUserTransform(transform)

            # Get ICP matrix
            update_transform(
                utils_3d.points_icp(
                    mesh_moving.points(),
                    mesh_fixed.points(),
                    threshold=icp_threshold,
                    scale=icp_scale,
                )
            )
        elif evt["keyPressed"] == "c":
            transform = vtk.vtkTransform()
            transform.SetMatrix(np.eye(4).flatten())
            mesh_fixed.SetUserTransform(transform)
        elif evt["keyPressed"] == "s":
            if tf is not None:
                matrix = get_mat(mesh_moving.GetUserTransform())
                LOG.debug("The optimal transformation matrix is:")
                LOG.debug(matrix)
                LOG.debug("Saving transform to: {}".format(tf))
                utils_3d.save_transform(tf, matrix)
                LOG.info("Transform saved")
            if output is not None:
                LOG.debug("Saving mesh to: {}".format(output))
                utils_3d.trimesh_transform_matrix(
                    input_mesh, get_mat(mesh_moving.GetUserTransform())
                ).export(output)
                LOG.info("Model saved")

    plt.addCallback("keyPressed", keypress_callback)

    plt.show(interactive=True)
    plt.closeWindow()

    if output is not None:
        if tf is not None:
            matrix = get_mat(mesh_moving.GetUserTransform())
            LOG.debug("The optimal transformation matrix is:")
            LOG.debug(matrix)
            LOG.debug("Saving transform to: {}".format(tf))
            utils_3d.save_transform(tf, matrix)
            LOG.info("Transform saved")
        if output is not None:
            LOG.debug("Saving mesh to: {}".format(output))
            utils_3d.trimesh_transform_matrix(
                input_mesh, get_mat(mesh_moving.GetUserTransform())
            ).export(output)
            LOG.info("Model saved")

    return utils_3d.trimesh_transform_matrix(
        input_mesh, get_mat(mesh_moving.GetUserTransform())
    ), get_mat(mesh_moving.GetUserTransform())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "fixed",
        help="Fixed reference mesh.",
    )
    parser.add_argument(
        "moving",
        help="Moving mesh, will be aligned to fixed mesh.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the resulting aligned mesh to.",
    )
    parser.add_argument(
        "--tf",
        default=None,
        help="Path to save the orientation data at. Must be a .json file.",
    )
    parser.add_argument(
        "--icp_scale",
        default=False,
        action="store_true",
        help="If ICP is allowed to scale.",
    )
    parser.add_argument(
        "--icp_threshold",
        default=75,
        help="ICP threshold for point correspondences.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    input_mesh = trimesh.load(args.moving)
    if args.fixed is not None:
        args.fixed = trimesh.load(args.fixed)
    register(
        moving=input_mesh,
        fixed=args.fixed,
        tf=args.tf,
        output=args.output,
        icp_threshold=args.icp_threshold,
        icp_scale=args.icp_scale,
    )
