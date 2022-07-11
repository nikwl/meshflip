import argparse
import logging

import trimesh
import pymeshfix

import meshflip.logger as logger


def repair_watertight(mesh):
    """Attempt to repair a mesh using the default pymeshfix procedure"""
    mesh = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    mesh.repair(joincomp=True, remove_smallest_components=False)
    return trimesh.Trimesh(mesh.v, mesh.f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "mesh",
        help="3D object to orient.",
    )
    parser.add_argument(
        "output",
        default=None,
        help="Path to save the transformed model at.",
    )
    logger.add_logger_args(parser)
    args = parser.parse_args()
    logger.configure_logging(args)

    logging.debug("Loading mesh from: {}".format(args.mesh))
    input_mesh = trimesh.load(args.mesh)
    repair_watertight(
        input_mesh,
    ).export(args.output)