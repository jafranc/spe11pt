import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import Data, Conversion


class Sparse_Data(Data):
    """ Class for handling from vtm time series to sparse data SPE11-CSP"""

    def __init__(self, simulator_name, version, solubility_file, units):
        super().__init__(simulator_name, version)

        self.converters = [('sec', 1), ('kg', 1), ('Pa', 1)]

        if 's' in units:
            self.converters[0] = ('sec', Conversion.SEC2SEC)
        elif 'h' in units:
            self.converters[0] = ('hour', Conversion.SEC2HOUR)
        elif 'd' in units:
            self.converters[0] = ('day', Conversion.SEC2DAY)
        elif 'y' in units:
            self.converters[0] = ('year', Conversion.SEC2YEAR)

        if 'kg' in units:
            self.converters[1] = ('kg', Conversion.KG2KG)
        elif 't' in units:
            self.converters[1] = ('t', Conversion.KG2T)
        elif 'g' in units:
            self.converters[1] = ('g', Conversion.KG2G)

        if 'Pa' in units:
            self.converters[2] = ('Pa', Conversion.PA2PA)
        elif 'bar' in units:
            self.converters[2] = ('bar', Conversion.PA2BAR)

        self.path_to_solubility = solubility_file
        self.sim_name = simulator_name

        # geom in meters
        if self.sim_name == "GEOS":
            self.boxes = {'Whole': [(0.0, 0.0, -1.2), (2.8, .01, 0.0)]}
            self.PO1 = [1.5, -0.7]
            self.PO2 = [1.7, -0.1]
            offx, offz = (0., 0.)
            # origin
            Ox, Oy, Oz = self.boxes['Whole'][0]
            # full length
            Lx, Ly, Lz = self.boxes['Whole'][1]
            Lx -= Ox
            Ly -= Oy
            Lz -= Oz

            self.boxes['A'] = [(Ox + 1.1 / 2.8 * Lx, Oy, Oz), (Ox + Lx - offx, Oy + Ly, Oz + 0.6 / 1.2 * Lz + offz)]
            self.boxes['B'] = [(Ox + offx, Oy, Oz + 0.6 / 1.2 * Lz + offz), (Ox + 1.1 / 2.8 * Lx, Oy + Ly, Oz + Lz + offz)]
            self.boxes['C'] = [(Ox + 1.1 / 2.8 * Lx, Oy, Oz + 0.1 / 1.2 * Lz + offz),
                               (Ox + 2.6 / 2.8 * Lx, Oy + Ly, Oz + 0.5 / 1.2 * Lz + offz)]

        elif self.sim_name == "OPM":
            self.boxes = {'Whole': [(0.0, 0.0, 1.2), (2.8, .01, 0.0)]}
            self.PO1 = [1.5, 0.7]
            self.PO2 = [1.7, 0.1]
            offx, offz = (0., 0.)
            # origin
            Ox, Oy, Oz = self.boxes['Whole'][0]
            # full length
            Lx, Ly, Lz = self.boxes['Whole'][1]
            Lx -= Ox
            Ly -= Oy
            Lz -= Oz

            self.boxes['A'] = [(Ox + 1.1 / 2.8 * Lx, Oy, Oz), (Ox + Lx - offx, Oy + Ly, Oz + 0.6 / 1.2 * Lz + offz)]
            self.boxes['B'] = [(Ox + offx, Oy, Oz + 0.6 / 1.2 * Lz + offz), (Ox + 1.1 / 2.8 * Lx, Oy + Ly, Oz + Lz + offz)]
            self.boxes['C'] = [(Ox + 1.1 / 2.8 * Lx, Oy, Oz + 0.1 / 1.2 * Lz + offz),
                               (Ox + 2.6 / 2.8 * Lx, Oy + Ly, Oz + 0.5 / 1.2 * Lz + offz)]

        self.schedule = np.arange(0, 5 * Conversion.SEC2DAY, 30000.)

        if self.version == 'b':
            for name in ['Whole', 'A', 'B', 'C'] :
                self.boxes[name] = [(item[0] * 3000, 1. / 0.01 * item[1], item[2] * 1000) for item in
                                   self.boxes[name]]
            self.PO1 = [self.PO1[0] * 3000, self.PO1[1] * 1000]
            self.PO2 = [self.PO2[0] * 3000, self.PO2[1] * 1000]
            self.schedule = np.arange(0., 1000 * Conversion.SEC2YEAR, 10 * Conversion.SEC2TENTHOFYEAR)
            self.schedule = np.arange(0., 700 * Conversion.SEC2YEAR, 50 * Conversion.SEC2YEAR)
            ## tmp for OPM
            # self.schedule = np.arange(1000* Conversion.SEC2YEAR, 2000*Conversion.SEC2YEAR, 5* Conversion.SEC2YEAR)
        elif self.version == 'c':
            self.schedule = np.arange(0., 1000 * Conversion.SEC2YEAR, 10 * Conversion.SEC2TENTHOFYEAR)

            # self.schedule = np.arange(0., 1000 * Conversion.SEC2YEAR, 1000 * Conversion.SEC2YEAR / 200)
            # self.schedule = np.arange(0., 615*Conversion.SEC2YEAR, 5*Conversion.SEC2YEAR)
            self.boxes['Whole'] = [(item[0] * 3000, 5000. / 0.01 * item[1], item[2] * 1000) for item in
                                   self.boxes['Whole']]
            self.boxes['Whole'][0] = (0., 0., 0.)
            self.PO1 = [self.PO1[0] * 3000, 2500, (self.PO1[1] + 1.2) * 1000]
            self.PO2 = [self.PO2[0] * 3000, 2500, (self.PO2[1] + 1.2) * 1000]
            offx, offz = (100., 150.)



    def process(self, directory, ifile, use_smry = False):

        super().process(directory, ifile)
        self._write_(directory, ifile, use_smry)
        self._plot_(directory)

    def _integrate_2_(self, pts, fields, box):
        """ Simplistic integration equivalent to Paraview integrate variables """
        xmin, ymin, zmin = ( min(box[0][0],box[1][0]),min(box[0][1],box[1][1]),min(box[0][2],box[1][2]) )
        xmax, ymax, zmax =  ( max(box[0][0],box[1][0]),max(box[0][1],box[1][1]),max(box[0][2],box[1][2]) )
        ii = np.intersect1d(np.intersect1d(np.argwhere((pts[:, 0] > xmin)), np.argwhere((pts[:, 0] < xmax))),
                            np.intersect1d(np.argwhere((pts[:, 2] > zmin)), np.argwhere((pts[:, 2] < zmax))))
        return fields[ii].sum()

    def _integrate_3_(self, pts, fields, box):
        """ Simplistic integration equivalent to Paraview integrate variables """
        xmin, ymin, zmin = ( min(box[0][0],box[1][0]),min(box[0][1],box[1][1]),min(box[0][2],box[1][2]) )
        xmax, ymax, zmax =  ( max(box[0][0],box[1][0]),max(box[0][1],box[1][1]),max(box[0][2],box[1][2]) )
        ii = np.intersect1d(
            np.intersect1d(np.intersect1d(np.argwhere((pts[:, 0] > xmin)), np.argwhere((pts[:, 0] < xmax))),
                           np.intersect1d(np.argwhere((pts[:, 2] > zmin)), np.argwhere((pts[:, 2] < zmax)))),
            np.intersect1d(np.argwhere((pts[:, 1] > ymin)), np.argwhere((pts[:, 1] < ymax)))
        )
        return fields[ii].sum()

    #
    def _integrate_gradient_2_(self, fn, vol, box, dims):
        """ Integrate gradient of fields re-interpolate on regular grid to deal with arbitrary mesh gradient """
        xmin, ymin, zmin = box[0]
        xmax, ymax, zmax = box[1]
        x, z = np.meshgrid(np.linspace(xmin, xmax, dims[0]), np.linspace(zmin, zmax, dims[1]))
        dx, dy, dz = (np.abs(xmax - xmin) / dims[0], np.abs(ymax - ymin) / 1, np.abs(zmax - zmin) / dims[1])
        res = fn(np.asarray([x.flatten(), z.flatten()]).transpose())
        dres = np.gradient(np.reshape(res, (dims[1], dims[0])),dx,dz)
        return np.sum(np.sqrt(np.square(dres[0]) + np.square(dres[1])))*dx*dy*dz

    def _integrate_gradient_3_(self, fn, vol, box, dims):
        """ Integrate gradient of fields re-interpolate on regular grid to deal with arbitrary mesh gradient """
        xmin, ymin, zmin = box[0]
        xmax, ymax, zmax = box[1]
        x, y, z = np.meshgrid(np.linspace(xmin, xmax, dims[0]), np.linspace(ymin, ymax, dims[1]),
                              np.linspace(zmin, zmax, dims[2]))
        dx, dy, dz = ((xmax - xmin) / dims[0], (ymax - ymin) / dims[1], (zmax - zmin) / dims[2])
        res = fn(np.asarray([x.flatten(), y.flatten(), z.flatten()]).transpose())
        res /= (vol(np.asarray([x.flatten(), y.flatten(), z.flatten()]).transpose()) + 1e-12)
        dres = np.gradient(np.reshape(res, (dims[0], dims[1], dims[2])))
        return (np.sqrt(np.power(dres[0], 2) + np.power(dres[1], 2) + np.power(dres[2], 2)) * dx * dy * dz).sum()


    def _thread_this_(self, ifile, olist_, ff, time):

        pts_from_vtk, fields = self._process_time_(ifile, time, olist=olist_)

        # some lines for MC magic number
        if self.version[0] == 'a' or self.version[0] == 'c':
            fields['mCO2Max'] = ff(fields['pres'], 293) * fields['rL']
        else:
            # convert it to kgCO2/m3Brine
            fields['mCO2Max'] = ff(fields['pres'], fields['temp']) * fields['rL']
        self.formula['M_C'] = 'mCO2/mCO2Max'

        for key, form in self.formula.items():
            fields[key] = self.process_keys(form, fields)

        if self.version[0] != "c":
            fn = self._get_interpolate_(pts_from_vtk,
                                        {'pres': fields['pres'], 'M_C': fields['M_C'], 'vol': fields['vol']})
        else:
            fn = self._get_interpolate_(pts_from_vtk,
                                        {'pres': fields['pres'], 'M_C': fields['M_C'], 'vol': fields['vol']},
                                        nskip=5)

        line = [time]
        # deal with P1 and P2
        p1 = fn['pres'](self.PO1)
        p2 = fn['pres'](self.PO2)
        line.extend([p1[0],p2[0]])
        # deal box A & B
        if self.version[0] != "c":
            for box_name, box in self.boxes.items():
                if box_name in ['A', 'B']:
                    line.extend([
                        self._integrate_2_(pts_from_vtk, fields['mMobile'], box),
                        self._integrate_2_(pts_from_vtk, fields['mImmobile'], box),
                        self._integrate_2_(pts_from_vtk, fields['mDissolved'], box),
                        self._integrate_2_(pts_from_vtk, fields['mSeal'], box)
                    ])
            # #deal box C
            line.append(
                self._integrate_gradient_2_(fn['M_C'], fn['vol'], self.boxes['C'], (1000, 500)) )
            # #deal sealTot
            line.append(
                self._integrate_2_(pts_from_vtk, fields['mSeal'], self.boxes['Whole']))
        else:
            for box_name, box in self.boxes.items():
                if box_name in ['A', 'B']:
                    line.extend([
                        self._integrate_3_(pts_from_vtk, fields['mMobile'], box),
                        self._integrate_3_(pts_from_vtk, fields['mImmobile'], box),
                        self._integrate_3_(pts_from_vtk, fields['mDissolved'], box),
                        self._integrate_3_(pts_from_vtk, fields['mSeal'], box)
                    ])
                # #deal box C
            line.append(
                    self._integrate_gradient_3_(fn['M_C'], fn['vol'], self.boxes['C'], (150, 10, 50)))
            # #deal sealTot
            line.append(self._integrate_3_(pts_from_vtk, fields['mSeal'], self.boxes['Whole']))

        return pd.DataFrame(data=[line],
                            columns= ['t[s]', 'p1[Pa]', 'p2[Pa]', 'mobA[kg]', 'immA[kg]', 'dissA[kg]', 'sealA[kg]',
                'mobB[kg]', 'immB[kg]', 'dissB[kg]', 'sealB[kg]', 'M_C[m]', 'sealTot[kg]'])

    def _write_(self, directory, ifile, use_smry : bool):
        from tqdm import tqdm
        # note that solubility is in KgCO2/KgBrine
        ff = self._process_solubility_(self.path_to_solubility)

        # preprocess input list from desired output
        if self.sim_name == "OPM":
            olist_ = set( self.name_indirection.keys() )
        elif self.sim_name == "GEOS":
            import re
            olist_ = set(
                [item for name in self.name_indirection.keys() for item in re.findall(r'(\w+)_\d|(\w+)', name)[0] if
                 len(item) > 0])

        # for time in tqdm(self.schedule):
        # import multiprocessing as mp
        # from functools import partial
        pdlist = []
        # pool = mp.Pool()
        # pdlist.append(pool.map(partial(self._thread_this_, ifile, olist_, ff), self.schedule))
        # pool.close()
        # pool.join()
        if use_smry:
            df = self._from_opm_rst_smry(ifile)
        else:
            for time in self.schedule:
                pdlist.append(self._thread_this_(ifile,olist_,ff,time))
            df = pd.concat(pdlist, ignore_index=True)



        #write off panda dataframe ordered by time
        df.sort_values(by=['t[s]'])
        df.to_csv('./' + directory + '/spe11' + self.version + '_time_series.csv')



    def _plot_(self, directory):

        import pandas as pd
        df = pd.read_csv('./' + directory + '/spe11' + self.version + '_time_series.csv')
        fig, axs = plt.subplots(2, 2)
        (time_name, time_unit), (mass_name, mass_unit), (pressure_name, pressure_unit) = self.converters
        # pressures
        axs[0][0].plot(df['t[s]'].to_numpy() / time_unit, df['p1[Pa]'].to_numpy() / pressure_unit,
                       label=f'pressure 1 [{pressure_name}]')
        axs[0][0].plot(df['t[s]'].to_numpy() / time_unit, df['p2[Pa]'].to_numpy() / pressure_unit,
                       label=f'pressure 2 [{pressure_name}]')
        axs[0][0].legend()
        # box A
        axs[0][1].plot(df['t[s]'].to_numpy() / time_unit, df['mobA[kg]'].to_numpy() / mass_unit,
                       label=f'mobile CO2 [{mass_name}]')
        axs[0][1].plot(df['t[s]'].to_numpy() / time_unit, df['immA[kg]'].to_numpy() / mass_unit,
                       label=f'immobile CO2 [{mass_name}]')
        axs[0][1].plot(df['t[s]'].to_numpy() / time_unit, df['dissA[kg]'].to_numpy() / mass_unit,
                       label=f'dissolved CO2 [{mass_name}]')
        axs[0][1].plot(df['t[s]'].to_numpy() / time_unit, df['sealA[kg]'].to_numpy() / mass_unit,
                       label=f'seal CO2 [{mass_name}]')
        axs[0][1].legend()
        axs[0][1].set_title('boxA')
        # box B
        axs[1][0].plot(df['t[s]'].to_numpy() / time_unit, df['mobB[kg]'].to_numpy() / mass_unit,
                       label=f'mobile CO2 [{mass_name}]')
        axs[1][0].plot(df['t[s]'].to_numpy() / time_unit, df['immB[kg]'].to_numpy() / mass_unit,
                       label=f'immobile CO2 [{mass_name}]')
        axs[1][0].plot(df['t[s]'].to_numpy() / time_unit, df['dissB[kg]'].to_numpy() / mass_unit,
                       label=f'dissolved CO2 [{mass_name}]')
        axs[1][0].plot(df['t[s]'].to_numpy() / time_unit, df['sealB[kg]'].to_numpy() / mass_unit,
                       label=f'seal CO2 [{mass_name}]')
        axs[1][0].legend()
        axs[1][0].set_title('boxB')
        # boxC
        axs[1][1].plot(df['t[s]'].to_numpy() / time_unit, df['M_C[m]'].to_numpy(), label='M_C[m]')
        axs[1][1].legend()
        axs[1][1].set_title('boxC')

        fig.savefig('./' + directory + '/spe11' + self.version + '_timeseries.png', bbox_inches='tight')


    def _from_opm_rst_smry(self,ifile) -> pd.DataFrame:
        # back up code from OPM repo

        from opm.io.ecl import EclFile as OpmFile
        from opm.io.ecl import EGrid as OpmGrid
        from opm.io.ecl import ERst as OpmRestart
        from opm.io.ecl import ESmry as OpmSummary
        base_pvd = ifile.split('.')[-2]
        dig = {}
        dig["unrst"] = OpmRestart(f"{base_pvd}.UNRST")
        dig["init"] = OpmFile(f"{base_pvd}.INIT")
        dig["egrid"] = OpmGrid(f"{base_pvd}.EGRID")
        gxyz = [
        dig["egrid"].dimension[0],
        dig["egrid"].dimension[1],
        dig["egrid"].dimension[2] ]
        dig["smspec"] =  OpmSummary(f"{base_pvd}.SMSPEC")
        dig["norst"] = len(dig["unrst"].report_steps)
        if dig["unrst"].count("WAT_DEN", 0):
            dig["watDen"], dig["r_s"], dig["r_v"] = "wat_den", "rsw", "rvw"
            dig["bpr"] = "BWPR"
        else:
            dig["watDen"], dig["r_s"], dig["r_v"] = "oil_den", "rs", "rv"
            dig["bpr"] = "BPR"
        dig["porv"] = np.array(dig["init"]["PORV"])
        dig["porva"] = np.array([porv for porv in dig["porv"] if porv > 0])
        ##
        dil = {'p1[Pa]': np.ndarray((dig["norst"],),dtype=float),
               'p2[Pa]':np.ndarray((dig["norst"],),dtype=float),
               'mobA[kg]':np.ndarray((dig["norst"],),dtype=float),
               'immA[kg]':np.ndarray((dig["norst"],),dtype=float),
               'dissA[kg]':np.ndarray((dig["norst"],),dtype=float),
               'sealA[kg]':np.ndarray((dig["norst"],),dtype=float),
               'mobB[kg]':np.ndarray((dig["norst"],),dtype=float),
               'immB[kg]':np.ndarray((dig["norst"],),dtype=float),
               'dissB[kg]':np.ndarray((dig["norst"],),dtype=float),
               'sealB[kg]':np.ndarray((dig["norst"],),dtype=float),
               'M_C[m]':np.ndarray((dig["norst"],),dtype=float),
               'sealTot[kg]':np.ndarray((dig["norst"],),dtype=float),
               'boundTot[kg]':np.ndarray((dig["norst"],),dtype=float)
               }
        fipnum = list(dig["init"]["FIPNUM"])

        dil['t[s]'] = np.linspace(0,np.max(self.schedule),dig["norst"])

        for istep,ti in enumerate(dil['t[s]']):
            pop1 = dig["unrst"]["PRESSURE", istep][fipnum.index(8)]
            pop2 = dig["unrst"]["PRESSURE", istep][fipnum.index(9)]
            if dig["unrst"].count("PCGW", istep):
                pop1 -= dig["unrst"]["PCGW", istep][fipnum.index(8)]
                pop2 -= dig["unrst"]["PCGW", istep][fipnum.index(9)]

            dil['p1[Pa]'][istep] = pop1 * 1.0e5# + list(dig["smspec"][names[sort[0]]] * 1.0e5)  # Pa
            dil['p2[Pa]'][istep] = pop2 * 1.0e5# + list(dig["smspec"][names[sort[1]]] * 1.0e5)  # Pa
            isubstep = np.argmin(np.abs((dig["smspec"]["TIME"]*Conversion.SEC2DAY)-ti))
            for i in [2, 4, 5, 8]:
                dil["mobA[kg]"][istep] += dig["smspec"][f"RGKDM:{i}"][isubstep] * Conversion.KMOL_TO_KG
                dil["immA[kg]"][istep] += dig["smspec"][f"RGKDI:{i}"][isubstep]* Conversion.KMOL_TO_KG
                dil["dissA[kg]"][istep] += dig["smspec"][f"RWCD:{i}"][isubstep]* Conversion.KMOL_TO_KG
            for i in [5, 8]:
                dil["sealA[kg]"][istep] += (
                    dig["smspec"][f"RWCD:{i}"][isubstep]
                    + dig["smspec"][f"RGKDM:{i}"][isubstep]
                    + dig["smspec"][f"RGKDI:{i}"][isubstep]
                ) * Conversion.KMOL_TO_KG
            for i in [3, 6]:
                dil["mobB[kg]"][istep] += dig["smspec"][f"RGKDM:{i}"][isubstep] * Conversion.KMOL_TO_KG
                dil["immB[kg]"][istep] += dig["smspec"][f"RGKDI:{i}"][isubstep] * Conversion.KMOL_TO_KG
                dil["dissB[kg]"][istep] += dig["smspec"][f"RWCD:{i}"][isubstep] * Conversion.KMOL_TO_KG
            for key in ["RWCD:6", "RGKDM:6", "RGKDI:6"]:
                dil["sealB[kg]"][istep] += dig["smspec"][key][isubstep] * Conversion.KMOL_TO_KG
            dil["sealTot[kg]"][istep] = dil["sealA[kg]"][istep] + dil["sealB[kg]"][istep]
            for name in ["RWCD", "RGKDM", "RGKDI"]:
                dil["sealTot[kg]"][istep] += (
                    dig["smspec"][f"{name}:7"][isubstep] + dig["smspec"][f"{name}:9"][isubstep]
                ) * Conversion.KMOL_TO_KG
            if self.version != "a":
                sealbound = (
                    dig["smspec"]["RWCD:10"][isubstep]
                    + dig["smspec"]["RGKDM:10"][isubstep]
                    + dig["smspec"]["RGKDI:10"][isubstep]
                ) * Conversion.KMOL_TO_KG
                dil["sealTot[kg]"][istep] += sealbound
                dil["boundTot[kg]"][istep] = (
                    sealbound
                    + (
                        dig["smspec"]["RWCD:11"][isubstep]
                        + dig["smspec"]["RGKDM:11"][isubstep]
                        + dig["smspec"]["RGKDI:11"][isubstep]
                    )
                    * Conversion.KMOL_TO_KG
                )

            self.compute_m_c(dig,dil,fipnum, gxyz, istep)

        return pd.DataFrame(dil)

    def compute_m_c(self,dig, dil, fipnum, gxyz, t_n):
        """Normalized total variation of the concentration field within Box C"""
        # snippets from opm - name changed - to be compared to vtk based
        dx,dy,dz = (dig["init"]["DX"],dig["init"]["DY"],dig["init"]["DZ"])
        boxc = np.array([fip in (4, 12) for fip in fipnum])
        boxc_x = np.roll(boxc, 1)
        boxc_y = np.roll(boxc, - gxyz[0])
        boxc_z = np.roll(boxc, -gxyz[0] * gxyz[1])
        xcw_max = 0
        sgas = abs(np.array(dig["unrst"]["SGAS", t_n]))
        rhow = np.array(dig["unrst"][f"{dig['watDen'].upper()}", t_n])
        rss = np.array(dig["unrst"][f"{dig['r_s'].upper()}", t_n])
        co2_d = rss * rhow * (1.0 - sgas) * dig["porva"] * Conversion.GAS_DEN_REF / Conversion.WAT_DEN_REF
        h2o_l = (1 - sgas) * rhow * dig["porva"]
        mliq = co2_d + h2o_l
        xco2 = 0.0 * co2_d
        inds = mliq > 0.0
        xco2[inds] = np.divide(co2_d[inds], mliq[inds])
        xcw = xco2
        #wait for xcwmax
        xcw_max = max(np.max(xco2[boxc]), xcw_max)
        if xcw_max != 0:
            xcw /= xcw_max
        if self.version != "c":
            dil["M_C[m]"][t_n] = np.sum(
                    np.abs(
                        (xcw[boxc_x] - xcw[boxc])
                        * dz[boxc]
                    )
                    + np.abs(
                        (xcw[boxc_z] - xcw[boxc])
                        * dx[boxc]
                    )
                )
        else:
            dil["M_C[m]"][t_n]= np.sum(
                    np.abs(
                        (xcw[boxc_x] - xcw[boxc])
                        * dy[boxc]
                        * dz[boxc]
                    )
                    + np.abs(
                        (xcw[boxc_y] - xcw[boxc])
                        * dil["dx"][boxc]
                        * dil["dz"][boxc]
                    )
                    + np.abs(
                        (xcw[boxc_z] - xcw[boxc])
                        * dil["dx"][boxc]
                        * dil["dy"][boxc]
                    )
                )
        return dil