import argparse
import logging

import vtk
import vedo
import trimesh
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


def orienter(
    input_mesh,
    static,
    output=None,
    tf=None,
    pca=False,
    center=False,
    normalize=False,
    pre_tf=None,
):

    # Instantiate the plotter
    plt = Plotter(
        N=4,
        sharecam=False,
    )
    plt += """
    Instructions: Adjust the object to be in its canonical orientation.
    Click and drag to move the the object. Use the middle mouse button to translate the object.
    Press "r" to reset the current transform.
    Press "n" to normalize the object.
    Press "c" to correct the fixed mesh (if you moved it by accident).
    Press "d" to drop the object.
    Press "s" to save.
    Press "q" to quit (and save).
    """

    # Load
    mesh_moving = utils_3d.trimesh2vedo(utils_3d.force_trimesh(input_mesh), c="r")
    meshes_static = []
    if static is not None:
        meshes_static = [
            utils_3d.trimesh2vedo(utils_3d.force_trimesh(m)) for m in static
        ]
    transform = vtk.vtkTransform()
    transform.SetMatrix(np.eye(4).flatten().flatten())
    for m in meshes_static:
        m.SetUserTransform(transform)
    if len(mesh_moving.faces()) == 0:
        mesh_moving = vedo.pointcloud.Points(mesh_moving.points())

    # Set a starting identity transform
    transform = vtk.vtkTransform()
    transform.SetMatrix(np.eye(4).flatten().flatten())
    mesh_moving.SetUserTransform(transform)

    def update_transform(mat):
        """Updates the user transform of the moving mesh"""
        mat = mat @ get_mat(mesh_moving.GetUserTransform())
        # assert np.isclose(np.linalg.det(mat[:3, :3]), 1, atol=1e-3)
        assert (mat[3, :] == np.array([0, 0, 0, 1])).all()
        transform = vtk.vtkTransform()
        transform.SetMatrix(mat.flatten())
        mesh_moving.SetUserTransform(transform)

    if pre_tf is not None:
        update_transform(pre_tf)
    if normalize:
        update_transform(
            utils_3d.trimesh_normalize_matrix(
                utils_3d.vedo2trimesh(mesh_moving), scale=True
            )
        )
    if center:
        update_transform(
            utils_3d.trimesh_normalize_matrix(utils_3d.vedo2trimesh(mesh_moving))
        )

    # Orient the moving mesh by its maximal dimension using PCA
    if pca:
        try:
            points = mesh_moving.to_trimesh().sample(100000)
        except (IndexError, AttributeError):
            points = mesh_moving.points()
        update_transform(utils_3d.points_maximal_orient(points))

    # Save this matrix for the reset button
    oriented_matrix = get_mat(mesh_moving.GetUserTransform())
    LOG.info("Loaded object successfully")

    plt.__prev_stable_mat = None
    plt.__stable_mat_ptr = 0

    def keypress_callback(evt):
        """
        Keypress callback
        """
        if evt["keyPressed"] == "r":
            LOG.info("Resetting ...")
            transform = vtk.vtkTransform()
            transform.SetMatrix(oriented_matrix.flatten())
            mesh_moving.SetUserTransform(transform)
            plt.window.Render()
            LOG.info("Object reset successfully")

        elif evt["keyPressed"] == "c":
            LOG.info("Correcting static transforms ...")
            transform = vtk.vtkTransform()
            transform.SetMatrix(np.eye(4).flatten())
            for m in meshes_static:
                m.SetUserTransform(transform)
            plt.window.Render()
            LOG.info("Objects corrected successfully")

        elif evt["keyPressed"] == "n":
            LOG.info("Normalizing object...")
            update_transform(
                utils_3d.trimesh_normalize_matrix(
                    utils_3d.vedo2trimesh(mesh_moving), scale=True
                )
            )
            LOG.info("Object normalized successfully")
            if center:
                LOG.info("Centering object ...")
                update_transform(
                    utils_3d.trimesh_normalize_matrix(utils_3d.vedo2trimesh(mesh_moving))
                )
                LOG.info("Object centered successfully")

        elif evt["keyPressed"] == "s":
            # If normalize and center were passed we need to rerun these
            if normalize:
                LOG.info("Normalizing object ...")
                update_transform(
                    utils_3d.trimesh_normalize_matrix(
                        utils_3d.vedo2trimesh(mesh_moving), scale=True
                    )
                )
            if center:
                LOG.info("Centering object ...")
                update_transform(
                    utils_3d.trimesh_normalize_matrix(utils_3d.vedo2trimesh(mesh_moving))
                )
                LOG.info("Object centered successfully")
            # Force the plotter to redraw the window
            plt.window.Render()
                
            if tf is not None:
                matrix = get_mat(mesh_moving.GetUserTransform())
                LOG.debug("The optimal transformation matrix is:")
                LOG.debug(matrix)
                LOG.debug("Saving transform to: {}".format(tf))
                utils_3d.save_transform(tf, matrix)
                LOG.info("Transform saved successfully")
            if output is not None:
                LOG.debug("Saving mesh to: {}".format(output))
                utils_3d.trimesh_transform_matrix(
                    input_mesh, get_mat(mesh_moving.GetUserTransform())
                ).export(output)
                LOG.info("Model saved successfully")

        elif evt["keyPressed"] == "d":
            LOG.info("Dropping object ...")
            if plt.__prev_stable_mat is not None:
                if np.array_equal(
                    plt.__prev_stable_mat, get_mat(mesh_moving.GetUserTransform())
                ):
                    plt.__stable_mat_ptr += 1
                else:
                    plt.__stable_mat_ptr = 0
            # Get a stable pose
            tfs, probs = utils_3d.vedo2trimesh(mesh_moving).compute_stable_poses()

            plt.__stable_mat_ptr = plt.__stable_mat_ptr % len(tfs)
            mat = tfs[plt.__stable_mat_ptr]

            LOG.info(
                "Displaying stable pose {} of {} with prob {:2.3f}".format(
                    plt.__stable_mat_ptr,
                    len(tfs),
                    probs[plt.__stable_mat_ptr],
                )
            )

            update_transform(mat)
            plt.__prev_stable_mat = get_mat(mesh_moving.GetUserTransform())

            # Force the plotter to redraw the window
            plt.window.Render()
            LOG.info("Model dropped successfully")

    plt.addCallback("keyPressed", keypress_callback)

    def udpate_camera(camera, pos, focal, up):
        camera.SetPosition(pos)
        camera.SetFocalPoint(focal)
        camera.SetViewUp(up)

    scale = 500

    # Head on
    plt.parallelProjection(at=0)
    udpate_camera(
        camera=plt.renderers[0].GetActiveCamera(),
        pos=(0, -scale, 0),
        focal=(0, 0, 0),
        up=(0, 0, 1),
    )
    plt.show(mesh_moving, title="Front view", at=0, mode=1, axes=1)
    for m in meshes_static:
        plt.show(m, title="Front view", at=0, mode=1, axes=1)

    # Left view
    plt.parallelProjection(at=1)
    udpate_camera(
        camera=plt.renderers[1].GetActiveCamera(),
        pos=(scale, 0, 0),
        focal=(0, 0, 0),
        up=(0, 0, 1),
    )
    plt.show(mesh_moving, at=1, mode=1, axes=1)
    for m in meshes_static:
        plt.show(m, title="Front view", at=1, mode=1, axes=1)

    # Down view
    plt.parallelProjection(at=2)
    udpate_camera(
        camera=plt.renderers[2].GetActiveCamera(),
        pos=(0, 0, scale),
        focal=(0, 0, 0),
        up=(0, 1, 0),
    )
    plt.show(mesh_moving, at=2, mode=1, axes=1)
    for m in meshes_static:
        plt.show(m, title="Front view", at=2, mode=1, axes=1)

    # Perspective view
    udpate_camera(
        camera=plt.renderers[3].GetActiveCamera(),
        pos=(scale, -scale, scale),
        focal=(0, 0, 0),
        up=(-1, 1, 1),
    )
    plt.show(mesh_moving, at=3, mode=1, axes=1)
    for m in meshes_static:
        plt.show(m, title="Front view", at=3, mode=1, axes=1)

    plt.show(interactive=True).close()

    if output is not None:
        # If normalize and center were passed we need to rerun these
        if normalize:
            update_transform(
                utils_3d.trimesh_normalize_matrix(
                    utils_3d.vedo2trimesh(mesh_moving), scale=True
                )
            )
        if center:
            update_transform(
                utils_3d.trimesh_normalize_matrix(utils_3d.vedo2trimesh(mesh_moving))
            )
            
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
        "input",
        type=str,
        help="Mesh to orient.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the resulting oriented model at.",
    )
    parser.add_argument(
        "--static",
        type=str,
        default=None,
        help="Static meshes to orient against.",
        nargs="+",
    )
    parser.add_argument(
        "--tf",
        type=str,
        default=None,
        help="Path to save the orientation data at. Must be a .json file.",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        default=False,
        help="If passed, will orient the object using pca.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        default=False,
        help="If passed, will center the object.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="If passed, will normalize the object. Note that this will also center the object.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    if args.static is not None:
        args.static = [trimesh.load(m) for m in args.static]
    LOG.debug("Loading mesh from: {}".format(args.input))
    args.input = trimesh.load(args.input)

    orienter(
        input_mesh=args.input,
        static=args.static,
        output=args.output,
        tf=args.tf,
        pca=args.pca,
        center=args.center,
        normalize=args.normalize,
    )
