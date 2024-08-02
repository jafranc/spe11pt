## Post-processing spe11 simulation

The SPE11-CSP project delivrable have specified format that require operations and batch processing of 
full output. This scripts help doing that.

A base abstract class `Data` is handling most of the field loading and interpolations that serves building of
remapped fields (a.k.a dense data in CSP words) and integrated results (a.k.a sparse data in CSP words).

Specialization classes `Sparse_Data` and `Dense_Data` are handling production of the data files and 
helper plots (representation of the data).

````python

   python3 main.py -o /path/to/out_spe11a --spe a --dense --pvd /path/to/vtkOutput.pvd --solute /path/to/solubility_table.csv --units h g Pa

````


````python

   python3 main.py -o /path/to/out_spe11b --spe b --sparse --pvd /path/to/vtkOutput.pvd  --units y t bar --solute /path/to/solubility_table.csv --units y kg bar

````

Similarly `Solver_Sparse` and `Solver_Dense` post-processes solver data from logs

````python

	python3 main.py -o /path/to/out_spe11b --spe b --sparsesolver --slurm /path/to/slurm-log.out --units y kg bar

````

_Note_ The 3d dense post-processing is not yet handled.
