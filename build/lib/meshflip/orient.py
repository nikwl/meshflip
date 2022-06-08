import argparse
import logging

import vtk
import vedo
import trimesh
import numpy as np
from vedo import Plotter

import mesflip.logger as logger
import mesflip.utils_3d as utils_3d


def get_mat(transform):
    """
    Get the matrix from a vtk transform object
    """
    return np.array([transform.GetMatrix().GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4, 4)


def orienter(input_mesh, output, savepath):

    # Instantiate the plotter
    plt = Plotter(
        N=4,
        sharecam=False,
    )
    plt += """
    Instructions: Adjust the object to be in its canonical orientation.
    Click and drag to move the the object. Use the middle mouse button to translate the object.
    Press "d" to drop the object.
    Press "s" to save the transformation to disk.
    Press "n" to normalize the object.
    Press "q" to quit.
    """

    # Load
    mesh_moving = utils_3d.trimesh2vedo(
        utils_3d.force_trimesh(
            input_mesh
        )
    )
    if len(mesh_moving.faces()) == 0:
        mesh_moving = vedo.pointcloud.Points(mesh_moving.points())

    # Set a starting identity transform
    transform = vtk.vtkTransform()
    transform.SetMatrix(np.eye(4).flatten().flatten())
    mesh_moving.SetUserTransform(transform)

    def update_transform(mat):
        """ Updates the user transform of the moving mesh """
        mat = mat @ get_mat(mesh_moving.GetUserTransform())
        # assert np.isclose(np.linalg.det(mat[:3, :3]), 1, atol=1e-3)
        assert (mat[3, :] == np.array([0, 0, 0, 1])).all()
        transform = vtk.vtkTransform()
        transform.SetMatrix(mat.flatten())
        mesh_moving.SetUserTransform(transform)

    # Center the moving mesh
    update_transform(
        utils_3d.trimesh_normalize_matrix(utils_3d.vedo2trimesh(mesh_moving))
    )

    # Orient the moving mesh by its maximal dimension using PCA
    try:
        points = mesh_moving.to_trimesh().sample(100000)
    except (IndexError, AttributeError):
        points = mesh_moving.points()
    update_transform(
        utils_3d.points_maximal_orient(points)
    )

    # Save this matrix for the reset button
    oriented_matrix = get_mat(mesh_moving.GetUserTransform())

    def keypress_callback(evt):
        """
        Keypress callback
        """
        if evt["keyPressed"] == "r":
            transform = vtk.vtkTransform()
            transform.SetMatrix(oriented_matrix.flatten())
            mesh_moving.SetUserTransform(transform)
            plt.window.Render()

        elif evt["keyPressed"] == "n":
            logging.info("Normalizing ...")
            update_transform(
                utils_3d.trimesh_normalize_matrix(utils_3d.vedo2trimesh(mesh_moving))
            )

        elif evt["keyPressed"] == "s":
            if savepath is not None:
                matrix = get_mat(mesh_moving.GetUserTransform())
                logging.debug("The optimal transformation matrix is:")
                logging.debug(matrix)
                logging.debug("Saving transform to: {}".format(savepath))
                utils_3d.save_transform(savepath, matrix)
                logging.info("Transform saved")
            if output is not None:
                logging.debug("Saving mesh to: {}".format(output))
                utils_3d.trimesh_transform_matrix(
                    input_mesh, 
                    get_mat(mesh_moving.GetUserTransform())
                ).export(output)
                logging.info("Model saved")
        
        elif evt["keyPressed"] == "d":
            _, mat = utils_3d.trimesh_settle(
                utils_3d.vedo2trimesh(mesh_moving),
                ransac_threshold=0.01,
                return_matrix=True,
            )
            update_transform(mat)
            plt.window.Render()

    plt.addCallback("keyPressed", keypress_callback)

    def udpate_camera(camera, pos, focal, up):
        camera.SetPosition(pos)
        camera.SetFocalPoint(focal)
        camera.SetViewUp(up)

    scale = 500

    # Head on
    plt.parallelProjection(at=0)
    udpate_camera(camera=plt.renderers[0].GetActiveCamera(), pos=(0,-scale,0), focal=(0,0,0), up=(0,0,1))
    plt.show(mesh_moving, title="Front view", at=0, mode=1, axes=1)
    
    # Left view
    plt.parallelProjection(at=1)
    udpate_camera(camera=plt.renderers[1].GetActiveCamera(), pos=(scale,0,0), focal=(0,0,0), up=(0,0,1))
    plt.show(mesh_moving, at=1, mode=1, axes=1)
    
    # Down view
    plt.parallelProjection(at=2)
    udpate_camera(camera=plt.renderers[2].GetActiveCamera(), pos=(0,0,scale), focal=(0,0,0), up=(0,1,0))
    plt.show(mesh_moving, at=2, mode=1, axes=1)
    
    # Perspective view
    udpate_camera(camera=plt.renderers[3].GetActiveCamera(), pos=(scale,-scale,scale), focal=(0,0,0), up=(-1,1,1))
    plt.show(mesh_moving, at=3, mode=1, axes=1)

    plt.show(interactive=True).close()

    if output is not None:
        if savepath is not None:
            matrix = get_mat(mesh_moving.GetUserTransform())
            logging.debug("The optimal transformation matrix is:")
            logging.debug(matrix)
            logging.debug("Saving transform to: {}".format(savepath))
            utils_3d.save_transform(savepath, matrix)
            logging.info("Transform saved")
        if output is not None:
            logging.debug("Saving mesh to: {}".format(output))
            utils_3d.trimesh_transform_matrix(
                input_mesh, 
                get_mat(mesh_moving.GetUserTransform())
            ).export(output)
            logging.info("Model saved")
    
    return utils_3d.trimesh_transform_matrix(
        input_mesh, 
        get_mat(mesh_moving.GetUserTransform())
    ), get_mat(mesh_moving.GetUserTransform())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "mesh",
        help="3D object to orient.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the transformed model at.",
    )
    parser.add_argument(
        "--savepath",
        default=None,
        help="Path to save the orientation data at. Must be a .json file.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    logging.debug("Loading mesh from: {}".format(args.mesh))
    input_mesh = trimesh.load(args.mesh)
    orienter(
        input_mesh, 
        args.output, 
        args.savepath,
    )