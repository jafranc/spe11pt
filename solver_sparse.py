import numpy as np
import pandas as pd

from solver import Solver, Conversion


class Solver_Sparse(Solver):

    def __init__(self, simulator_name, version, units):
        super().__init__(simulator_name, version)
        self.schedule = np.arange(0, 5 * Conversion.SEC2DAY, 3000.)
        if self.version == 'b':
            self.schedule = np.arange(0., 1000 * Conversion.SEC2YEAR, 1000 * Conversion.SEC2YEAR / 80)
        elif self.version == 'c':
            self.schedule = range(0, 50 * Conversion.SEC2YEAR, 5 * Conversion.SEC2YEAR)
            self.schedule.extend(75 * Conversion.SEC2YEAR, 100 * Conversion.SEC2YEAR)
            self.schedule.extend(range(100 * Conversion.SEC2YEAR, 500 * Conversion.SEC2YEAR, 50 * Conversion.SEC2YEAR))
            self.schedule.extend(
                range(500 * Conversion.SEC2YEAR, 1000 * Conversion.SEC2YEAR, 100 * Conversion.SEC2YEAR))

        self.converters = [('sec', 1), ('kg', 1), ('Pa', 1)]

        if 's' in units:
            self.converters[0] = ('sec', Conversion.SEC2SEC)
        elif 'h' in units:
            self.converters[0] = ('hour', Conversion.SEC2HOUR)
        elif 'd' in units:
            self.converters[0] = ('day', Conversion.SEC2DAY)
        elif 'y' in units:
            self.converters[0] = ('year', Conversion.SEC2YEAR)

    def process(self, directory, ifile):
        df, metadata = super().process(directory, ifile)
        self._write_(metadata, df, directory)
        self._plot_(directory)

    def _write_(self, header, df, directory):
        from tqdm import tqdm
        with open('./' + directory + '/spe11' + self.version + '_performance_time_series.csv', 'w') as f:
            f.write("#t[s], tstep[s], fsteps[-], mass[kg], dof[-], nliter[-], nres[-], "
                    "liniter[-], runtime[s], tlinsol[s]\n")

            t0 = self.schedule
            df = df[df['time'] > 0]
            for i in tqdm(range(len(self.schedule) - 1)):
                df_ = df[(df['time'] >= t0[i]) & (df['time'] < t0[i + 1])]
                f.write("{:3e}, {:3e}, {:3e}, {:3e}, {:3e}, {:3e}, {:3e}, {:3e}, {:3e}, {:3e}\n".format(
                    t0[i + 1], df_['dt'].mean(), df_['cut'].sum(), -1., header['ndof'], df_['nnl'].sum(), -1.,
                    df_['Iterations'].sum(),
                    df_['tsetup'].sum() + df_['tsolve'].sum(), df_['tsolve'].sum()
                ))

    def _plot_(self, directory):
        import matplotlib.pyplot as plt
        df = pd.read_csv('./' + directory + '/spe11' + self.version + '_performance_time_series.csv')
        fig, axs = plt.subplots(2, 2)
        fig.tight_layout()
        (time_name, time_unit), (_, _), (_, _) = self.converters

        axs[0][0].plot(df['#t[s]'].to_numpy() / time_unit, df[' runtime[s]'].to_numpy(), label='runtime[s]')
        axs[0][0].plot(df['#t[s]'].to_numpy() / time_unit, df[' tlinsol[s]'].to_numpy(), label='tlinsol[s]')
        axs[0][0].legend()
        axs[0][0].set_title('runtime')
        #
        axs[0][1].plot(df['#t[s]'].to_numpy() / time_unit, df[' nliter[-]'].to_numpy(), label='runtime[s]')
        axs[0][1].legend()
        axs[0][1].set_title('non-linear')
        #
        axs[1][0].plot(df['#t[s]'].to_numpy() / time_unit, df[' liniter[-]'].to_numpy(), label='liniter[-]')
        axs[1][0].twinx().bar(df['#t[s]'].to_numpy() / time_unit, df[' fsteps[-]'].to_numpy(), 10 , label='fsteps[-]',color='tab:orange')
        axs[1][0].set_title('linear')
        #
        axs[1][1].plot( df['#t[s]'].to_numpy() / time_unit, df[' tstep[s]'].to_numpy()/time_unit, label=f'tstep[{time_name}]')
        axs[1][1].legend()
        axs[1][1].set_title('time-step size stats')

        fig.savefig('./' + directory + '/spe11' + self.version + '_preformance_timeseries.png', bbox_inches='tight')
