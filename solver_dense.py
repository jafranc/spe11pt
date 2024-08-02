from solver import Solver,Conversion
import numpy as np

class Solver_Dense(Solver):

    def _write_(self, time, fn, directory):

        file_name = './' + directory + '/performance_spatial_map_' + "{time:2}".format(time=time / Conversion.SEC2HOUR) + 'h.csv'
        file_header = ("#x[m], y[m], z[m], cvol[m2], arat[-], CO2 max_norm_res[-], "
                       "H2O max_norm_res[-], CO2 mb_error[-], post est [-]\n")
        file_fmt = "%.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e"
        xyz = np.asarray([self.x.flatten(), self.z.flatten()]).transpose()
        output = np.asarray([])
