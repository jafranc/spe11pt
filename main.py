import argparse
from argparse import ArgumentParser

from dense_data import Dense_Data
from sparse_data import Sparse_Data
from solver_sparse import Solver_Sparse

if __name__ == "__main__":
    descr = 'Set of python script for post-processing spe11 from vtk files\n'
    parser: ArgumentParser = argparse.ArgumentParser(description=descr)

    # i/o
    parser.add_argument("--pvd",
                        help="path to the pvd dict file",
                        nargs=1, required=False)
    parser.add_argument("--slurm",
                        help="path to the log file",
                        nargs=1, required=False)
    parser.add_argument("--on-pvd",
                        help="path to the log file",
                        action='store_true', required=False)
    parser.add_argument("--solute",
                        help="path to the solubility csv file",
                        nargs=1, required=False)
    parser.add_argument("--units",
                        help="choose time,mass,pressure units for sparse data\nSymbols are s,d,y;g,kg,t;Pa,bar",
                        nargs=3)
    parser.add_argument("-o",
                        help="path to the output dir",
                        nargs=1, required=True)
    # version
    parser.add_argument("--spe",
                        help="version spe11 a,b or c",
                        nargs=1, required=True)
    parser.add_argument("--sim",
                        help="simulator name [GEOS, OPM]",
                        nargs=1, required=True)
    parser.add_argument("--smry",
                        help="wheter to use smry for OPM",
                        action="store_true")


    # what to do
    parser.add_argument("--sparse",
                        action='store_true',
                        help='build sparse data')

    parser.add_argument('--dense', action='store_true',
                        help='build colormaps from pvd')

    parser.add_argument('--sparsesolver', action='store_true',
                        help='build solver perf')

    args = parser.parse_args()

    if args.dense and args.pvd and args.solute:
        dense = Dense_Data(args.sim[0], args.spe, args.solute[0], on_pvd=args.on_pvd)
        dense.process(directory=args.o[0], ifile=args.pvd[0])
    elif args.sparse and args.pvd and args.solute and args.units:
        sparse = Sparse_Data(args.sim[0], args.spe, solubility_file=args.solute[0], units=args.units, on_pvd=args.on_pvd)
        sparse.process(directory=args.o[0], ifile=args.pvd[0], use_smry=args.smry)
    elif args.sparsesolver and args.slurm and args.units:
        sparse = Solver_Sparse(args.sim[0], args.spe, units=args.units)
        sparse.process(directory=args.o[0], ifile=args.slurm[0])
    else:
        raise NotImplemented()
