import os
import typing

import pandas as pd
import scipy
import numpy as np
from numpy.lib.function_base import meshgrid
from numpy.ma.core import shape
from scipy.optimize import direct
import argparse
from argparse import ArgumentParser
import gmsh

PHYSICAL_INDEX_OUTSIDE_OF_DOMAIN = 1000
SIZE_DOMAIN_Y = 5000.0
_model_name = ''
_2d_model_name = ''


class PhysicalIndexMapper:
    def __init__(self) -> None:
        gmsh.initialize()

        gmsh.open('/tmp/spe11c.geo')
        self._model_name = gmsh.model.getCurrent()
        self._variant = "C"

        gmsh.open('/tmp/spe11b.geo')
        self._2d_model_name = gmsh.model.getCurrent()

        gmsh.model.setCurrent(self._model_name)
        self._physical_groups = self._with_model_for_physical_index_queries(
            lambda: self._read_physical_groups()
        )

    def physical_index(self, position: tuple) -> int:
        position = self._project_to_model_for_index_queries(position)
        index = self._with_model_for_physical_index_queries(
            lambda: self._get_physical_index(position)
        )
        return PHYSICAL_INDEX_OUTSIDE_OF_DOMAIN if index is None else index

    def physical_groups(self, dim: int) -> dict:
        return {
            self._get_physical_name(dim, tag): tag
            for _, tag in self._physical_groups
        }

    def _read_physical_groups(self) -> dict:
        return {
            (d, t): gmsh.model.getEntitiesForPhysicalGroup(dim=d, tag=t)
            for d, t in gmsh.model.getPhysicalGroups()
        }

    def _get_physical_name(self, dim: int, tag: int) -> str:
        name = gmsh.model.getPhysicalName(dim, tag)
        return name if name else str(tag)


    def _with_model_for_physical_index_queries(self, action):
        if self._variant == "C":
            gmsh.model.setCurrent(self._2d_model_name)
        result = action()
        if self._variant == "C":
            gmsh.model.setCurrent(self._model_name)
        return result


    def _project_to_model_for_index_queries(self, position: tuple) -> tuple:
        if self._variant == "C":
            return (
                position[0],
                position[2] - z_offset_at(position[1]),
                0.0
            )
        return position


    def _get_physical_index(self, position: tuple) -> typing.Optional[int]:
        for dim, tag in gmsh.model.getEntities(self._query_model_dimension()):
            min, max = _get_bounding_box(gmsh.model, dim, tag)
            if _is_in_bbox(position, min, max):
                if gmsh.model.isInside(dim, tag, position):
                    return self._get_entity_physical_index(dim, tag)
        return None


    def _get_entity_physical_index(self, dim: int, tag: int) -> typing.Optional[int]:
        for (physical_dim, physical_index), entity_tags in self._physical_groups.items():
            if physical_dim == dim and tag in entity_tags:
                return physical_index
        return None


    def _model_dimension(self) -> int:
        return 3 if self._variant == "C" else 2


    def _query_model_dimension(self) -> int:
        return 2


def _is_in_bbox(position, min, max) -> bool:
    return all(position[i] <= max[i] and position[i] >= min[i] for i in range(3))


def _get_bounding_box(model, entity_dim=-1, entity_tag=-1) -> tuple:
    bbox = model.getBoundingBox(entity_dim, entity_tag)
    return tuple(bbox[:3]), tuple(bbox[3:])

# ##

# def physical_index(position: tuple) -> int:
#     gmsh.initialize()
#     gmsh.open('/work/data/geos/MAELSTROM/usecases/Jacques/spe11/meshes/org/spe11b.geo')
#     _model_name = gmsh.model.getCurrent()
#     _2d_model_name = gmsh.model.getCurrent()
#
#     position = _project_to_model_for_index_queries(position)
#     index = _with_model_for_physical_index_queries(
#         lambda: _get_physical_index(position)
#     )
#     return PHYSICAL_INDEX_OUTSIDE_OF_DOMAIN if index is None else index
#
#
# def _get_physical_index(position: tuple) -> typing.Optional[int]:
#     for dim, tag in gmsh.model.getEntities(2):
#         min, max = _get_bounding_box(gmsh.model, dim, tag)
#         if _is_in_bbox(position, min, max):
#             if gmsh.model.isInside(dim, tag, position):
#                 return _get_entity_physical_index(dim, tag)
#
#
# def _get_entity_physical_index(dim: int, tag: int) -> typing.Optional[int]:
#     for (physical_dim, physical_index), entity_tags in _physical_groups.items():
#         if physical_dim == dim and tag in entity_tags:
#             return physical_index
#     return None
#
#
# def _with_model_for_physical_index_queries(action):
#     gmsh.model.setCurrent(_2d_model_name)
#     result = action()
#     gmsh.model.setCurrent(_model_name)
#     return result
#
#
# def _project_to_model_for_index_queries(position: tuple) -> tuple:
#     return (
#         position[0],
#         position[2] - z_offset_at(position[1]),
#         0.0
#     )
#

def z_offset_at(y: float) -> float:
    """
    Compute the difference in z-coordinate between reference and physical space
    according to eq. (4.1) of the description for SPE11 Version C, given a
    y-coordinate in the reference space.
    """
    f = (y - 2500.0) / 2500.0
    return 150.0 * (1.0 - f * f) + 10.0 * y / SIZE_DOMAIN_Y


def get_interpolate(points_from_vtk, val, key, nskip=1):
    """ getting dict of proper interpolation for fields """
    print(f'Building interpolation for {key}')
    # do not interpolate if not interested by (and only brought as field is another component of
    # a field of interest
    T = points_from_vtk
    return get_lambda_3(T[::nskip, :], val[::nskip])


def get_lambda_3(pts, disc_func):
    # try NearestNDInterpolator(...)
    return scipy.interpolate.NearestNDInterpolator(pts, disc_func)
    # return scipy.interpolate.LinearNDInterpolator(pts, disc_func)


def read(directory, ifile, pts, name):
    # pts_from_vtk
    df = pd.read_csv(f'{directory}/{ifile}', usecols=name)
    return get_interpolate(pts, df[name].to_numpy(), name)


def write(directory, ofile, field_names, fn, pmapper : PhysicalIndexMapper, offset, scale=1):
    # reporting grid
    file_header = f'#x[m], y[m], z[m]'
    file_fmt = "%.3e, %.3e, %.3e"
    Nx, Ny, Nz = (168, 100, 120)
    x, y, z = np.meshgrid(np.linspace(0, 8400, Nx),
                          np.linspace(0, 5000, Ny),
                          np.linspace(0, 1200, Nz),
                          indexing='xy')
    xyz = np.asarray([x.flatten(), y.flatten(), z.flatten()]).transpose()
    for name in field_names:
        print(f'Writing interpolation for {name}')
        file_header += f', {name}'
        file_fmt += ", %.3e"

    output = np.concatenate((np.reshape(x.flatten(), (168 * 100 * 120,1)),
                         np.reshape(y.flatten(), (168 * 100 * 120,1)),
                         np.reshape(z.flatten(), (168 * 100 * 120,1)),
                         np.reshape(fn(xyz), (168 * 100 * 120, len(field_names)))), axis=1 )

    #complete output
    for ii in range(output.shape[0]):
        x,y,z=output[ii,:3]
        if pmapper.physical_index((x,y,z)) == PHYSICAL_INDEX_OUTSIDE_OF_DOMAIN:
            output[ii,3:] = np.nan

    np.savetxt(f'{directory}/{ofile}', output, fmt=file_fmt, delimiter=',', header=file_header)


if __name__ == "__main__":
    print('Main')
    # args processsing
    descr = 'converter from Paraview CSV to Bernd CSV'
    parser: ArgumentParser = argparse.ArgumentParser(description=descr)

    # i/o
    parser.add_argument("--path",
                        help="absolute directory path",
                        nargs=1, required=True)
    parser.add_argument("--root",
                        help="root of generated filenames",
                        nargs=1, required=True)

args = parser.parse_args()

# directory and root
if (args.path and args.root):
    schedule = range(0,26)  # directly in years
    pmapper = PhysicalIndexMapper()
    for index in schedule:
        df = pd.read_csv(f'{args.path[0]}/{args.root[0]}_{index}.csv',
                         usecols=['elementCenter:0', 'elementCenter:1', 'elementCenter:2'])
        pts_from_vtk = df.to_numpy()
        field_names = ["pressure", "phaseVolumeFraction:0", "fluid_phaseCompFraction:2","fluid_phaseCompFraction:1","fluid_phaseMassDensity:0","fluid_phaseMassDensity:1","compAmount:0", "temperatureC"]
        offset_ = {"pressure":0, "phaseVolumeFraction:0":0, "fluid_phaseCompFraction:2":0,"fluid_phaseCompFraction:1":0,"fluid_phaseMassDensity:0":0,"fluid_phaseMassDensity:1":0,"compAmount:0":0, "temperatureC":0}
        # for name in field_names:
        f = read(args.path[0], f'{args.root[0]}_{index}.csv', pts_from_vtk, field_names)
        write('/media/jfranc/Seagate Portable Drive/data-spe11/c/', f'{args.root[0]}_{index}_formatted.csv', field_names, f, pmapper, offset_)
