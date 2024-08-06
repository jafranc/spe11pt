from abc import ABCMeta, abstractmethod

import numpy as np
import os
import scipy


class Conversion:
    # sched in sec
    SEC2SEC = 1
    SEC2MIN = 60
    SEC2HOUR = 60*SEC2MIN
    SEC2DAY = 24 * 60 * SEC2MIN
    SEC2YEAR = 3.1536E7
    DAY2YEAR = 365
    SEC2TENTHOFYEAR = 3.1536E6
    KG2KG = 1
    KG2G = 1e-3
    KG2T = 1e3

    PA2PA = 1
    PA2BAR = 1e5

    CO2MOL2KG = 44.01e-3
    #opm
    KMOL_TO_KG = 0.044e3
    GAS_DEN_REF = 1.86843
    WAT_DEN_REF = 998.108

    K2C = 273.15

class Data(metaclass=ABCMeta):
    def __init__(self, simulator_name, version):
        self.use_smry = False
        self.version = version[0]
        self.seal_facies_tag = "reservoir1"
        self.data_sets = {}

        # default is a
        self.schedule = np.arange(0, 5 * Conversion.SEC2DAY, 0.5 * Conversion.SEC2DAY)
        # for GEOS
        if simulator_name == "GEOS":
            self.name_indirection = {'pressure': 'pres',
                                     'phaseVolumeFraction_0': 'satg',
                                     'phaseVolumeFraction_1': 'satw',
                                     'fluid_phaseCompFraction_2': 'mCO2',
                                     'fluid_phaseCompFraction_1': 'mH2O',
                                     'fluid_phaseDensity_0': 'rG',
                                     'fluid_phaseDensity_1': 'rL',
                                     'rockPorosity_porosity': 'poro',
                                     'elementVolume': 'vol',
                                     'phaseMobility_0': 'krg',
                                     'phaseMobility_1': 'krw'}

            self.formula = {'mImmobile': 'if(krg<8000, rG*satg*poro*vol)',
                            'mMobile': 'if(krg>8000,rG*satg*poro*vol)',
                            'mDissolved': 'rL*mCO2*poro*vol*satw',
                            'mSeal': 'if(sealtag > 0.0, rL*mCO2*poro*vol*satw + rG*satg*poro*vol)',
                            'mTotal': 'if(vol>5e4, rL*mCO2*poro*vol*satw + rG*poro*vol*satg)'}

        elif simulator_name == "OPM":
            self.name_indirection = { 'pressure_water': 'pres',
                                     'saturation_gas': 'satg',
                                     'saturation_water': 'satw',
                                     'moleFrac_gas^Gas': 'mCO2',
                                     'moleFrac_water^Water': 'mH2O',
                                     'density_gas': 'rG',
                                     'density_water': 'rL',
                                     'porosity': 'poro',
                                     'relativePerm_gas': 'krg',
                                     'relativePerm_water': 'krw'}
            # vol will be computed if not present
            self.formula = {'mImmobile': 'if(krg<0.001, rG*satg*poro*vol)',
                            'mMobile': 'if(krg>0.001,rG*satg*poro*vol)',
                            'mDissolved': 'rL*mCO2*poro*vol*satw',
                            'mSeal': 'if(sealtag > 0.0, rL*mCO2*poro*vol*satw + rG*satg*poro*vol)',
                            'mTotal': 'rL*mCO2*poro*vol*satw + rG*poro*vol*satg'}


        if version[0] in ['b', 'c']:
            # as described
            self.name_indirection['temperature'] = 'temp'

    def _get_filename_(self, pvdfile, time):
        if len(self.data_sets) == 0:
            self._read_pvd_(pvdfile)
        return os.path.dirname(pvdfile) + '/' + self.data_sets[time]

    def _read_pvd_(self, ifile):
        import xml.etree.ElementTree as ET
        tree = ET.parse(ifile)
        root = tree.getroot()
        for ds in root.find('Collection').findall('DataSet'):
            self.data_sets[float(ds.attrib['timestep'])] = ds.attrib['file']

    def _get_interpolate_(self, points_from_vtk, fields: dict, nskip=1):
        """ getting dict of proper interpolation for fields """
        fn = {}
        for key, val in fields.items():
            if key in self.name_indirection.values() or key in self.formula.keys():
                # do not interpolate if not interested by (and only brought as field is another component of
                # a field of interest
                T = points_from_vtk
                y = np.unique(T[:, 1])
                if y.shape[0] == 1:
                    fn[key] = self._get_lambda_2_((T[::nskip, 0], T[::nskip, 2]), val[::nskip])
                else:
                    fn[key] = self._get_lambda_3_((T[::nskip, 0], T[::nskip, 1], T[::nskip, 2]), val[::nskip])

        return fn

    def _get_lambda_2_(self, xpts, disc_func):
        x, z = xpts
        return scipy.interpolate.LinearNDInterpolator(np.asarray([x, z]).transpose(), disc_func, fill_value=0.0)

    def _get_lambda_3_(self, xpts, disc_func):
        x, y, z = xpts
        #try NearestNDInterpolator(...)
        return scipy.interpolate.NearestNDInterpolator(np.asarray([x, y, z]).transpose(), disc_func)
        # return scipy.interpolate.LinearNDInterpolator(np.asarray([x, y, z]).transpose(), disc_func, fill_value=0.0)

    def bounding_box(self,pvdfile):
        import vtk
        if self._get_filename_(pvdfile, 0).split('.')[-1] == 'vtm':
            reader = vtk.vtkXMLMultiBlockDataReader()
            reader.SetFileName(self._get_filename_(pvdfile, 0))
            reader.Update()

            it = reader.GetOutput().NewIterator()
            it.InitTraversal()
            _xm,_xM,_ym,_yM,_zm,_zM = (0.,0.,0.,0.,0.,0.)
            while not it.IsDoneWithTraversal():
                xm,xM,ym,yM,zm,zM = reader.GetOutput().GetDataSet(it).GetBounds()
                _xm = min(xm,_xm)
                _xM = max(xM,_xM)
                _ym = min(ym,_ym)
                _yM = max(yM,_yM)
                _zm = min(zm,_zm)
                _zM = max(zM,_zM)
                it.GoToNextItem()
        else:
            raise NotImplemented()

        return (_xm,_xM,_ym,_yM,_zm,_zM )

    def _process_time_(self, pvdfile, time, olist):

        import vtk
        import numpy as np
        from vtk.util import numpy_support

        if self._get_filename_(pvdfile, time).split('.')[-1] == 'vtm':
            reader = vtk.vtkXMLMultiBlockDataReader()
            reader.SetFileName(self._get_filename_(pvdfile, time))
            reader.Update()
            # remove well blocks - in hierarchy mesh.lvl0.CellElementRegion/WellElementRegion
            if reader.GetOutput().GetBlock(0).GetBlock(0).GetNumberOfBlocks() > 1:
                reader.GetOutput().GetBlock(0).GetBlock(0).RemoveBlock(1)

            # browse blocks - assemble size
            # counting cells and blocks/rank
            nv = reader.GetOutput().GetNumberOfCells()
            it = reader.GetOutput().NewIterator()
            it.InitTraversal()

            ncnf = 0
            it = reader.GetOutput().NewIterator()
            it.InitTraversal()
            for ifields in olist:
                ncc = reader.GetOutput().GetDataSet(it).GetCellData().GetArray(ifields).GetNumberOfComponents()
                ncnf += ncc
                it.GoToNextItem()

            # form data container
            f = np.zeros(shape=(nv, ncnf), dtype='float')
            seal_tag = np.zeros(shape=(nv, 1), dtype='float')
            pts = np.zeros(shape=(nv, 3), dtype='float')

            # get seal flagged
            mesh = reader.GetOutput().GetBlock(0).GetBlock(0).GetBlock(0)
            for i in range(0, mesh.GetNumberOfBlocks()):
                start = 0
                it = mesh.GetBlock(i).NewIterator()
                it.InitTraversal()
                info = mesh.GetMetaData(i)
                block_tag = info.Get(vtk.vtkMultiBlockDataSet.NAME())
                while not it.IsDoneWithTraversal():
                    field = mesh.GetBlock(i).GetDataSet(it).GetCellData().GetArray("ghostRank")
                    nt = field.GetNumberOfValues()
                    if self.seal_facies_tag == block_tag:
                        seal_tag[start:(start + nt), 0] = 1
                    it.GoToNextItem()

                nc = 0
                j = 0  # for field loop
                fielddict = {}
                for ifields in olist:
                    start = 0
                    it = reader.GetOutput().NewIterator()
                    it.InitTraversal()
                    while not it.IsDoneWithTraversal():
                        if ifields == 'relperm1_phaseRelPerm':
                            field = self._process_partial_('relperm', 'phaseRelPerm',
                                                           reader.GetOutput().GetDataSet(it).GetCellData(), 1,
                                                           8)
                        else:
                            field = reader.GetOutput().GetDataSet(it).GetCellData().GetArray(ifields)

                        nt = field.GetNumberOfValues()
                        nc = field.GetNumberOfComponents()
                        f[start:(start + int(nt / nc)), j:j + nc] = numpy_support.vtk_to_numpy(field).reshape(int(nt / nc),
                                                                                                              nc)

                        cc = vtk.vtkCellCenters()
                        cc.SetInputData(reader.GetOutput().GetDataSet(it))
                        cc.Update()
                        if nt > 0:
                            pts[start:(start + int(nt / nc)), :] = numpy_support.vtk_to_numpy(
                                cc.GetOutput().GetPoints().GetData())
                        it.GoToNextItem()
                        start += int(nt / nc)

                    if nc > 1:
                        for k in range(0, nc):
                            if ifields + "_" + str(k) in self.name_indirection:
                                fielddict[self.name_indirection[ifields + "_" + str(k)]] = f[:, j + k]
                    else:
                        if ifields in self.name_indirection:
                            fielddict[self.name_indirection[ifields]] = f[:, j]
                    j = j + nc

            fielddict['sealtag'] = seal_tag[:, 0]

        elif self._get_filename_(pvdfile, time).split('.')[-1] == 'vtu' or  self._get_filename_(pvdfile, time).split('.')[-1] == 'pvtu':

            # most likely OPM path
            if self._get_filename_(pvdfile, time).split('.')[-1] == 'pvtu':
                reader = vtk.vtkXMLPUnstructuredGridReader()
            elif  self._get_filename_(pvdfile, time).split('.')[-1] == 'vtu':
                reader = vtk.vtkXMLUnstructuredGridReader()

            reader.SetFileName(self._get_filename_(pvdfile, time))
            reader.Update()

            # browse blocks - assemble size
            # counting cells and blocks/rank
            nv = reader.GetOutput().GetNumberOfCells()
            mesh = reader.GetOutput()
            fielddict = {}
            ncnf = 0
            for ifields in olist:
                if  reader.GetOutput().GetCellData().GetArray(ifields) is not None :
                    ncc = reader.GetOutput().GetCellData().GetArray(ifields).GetNumberOfComponents()
                    ncnf += ncc
                else:
                    raise LookupError(f"{ifields} not known in {[reader.GetOutput().GetCellData().GetArrayName(i) for i in range(reader.GetOutput().GetCellData().GetNumberOfArrays())]}")

            mQ = vtk.vtkMeshQuality()
            mQ.SetHexQualityMeasureToVolume()
            mQ.SetInputData(reader.GetOutput())
            mQ.Update()
            mesh = mQ.GetOutput()
            self.name_indirection['Quality'] = 'vol'
            fielddict['vol'] = numpy_support.vtk_to_numpy(mesh.GetCellData().GetArray('Quality'))

                # form data container
            pts = np.zeros(shape=(nv, 3), dtype='float')

            # # get seal flagged
            # for i in range(0, mesh.GetNumberOfBlocks()):
            #     start = 0
            #     it = mesh.GetBlock(i).NewIterator()
            #     it.InitTraversal()
            #     info = mesh.GetMetaData(i)
            #     block_tag = info.Get(vtk.vtkMultiBlockDataSet.NAME())
            #     while not it.IsDoneWithTraversal():
            #         field = mesh.GetBlock(i).GetDataSet(it).GetCellData().GetArray("ghostRank")
            #         nt = field.GetNumberOfValues()
            #         if self.seal_facies_tag == block_tag:
            #             seal_tag[start:(start + nt), 0] = 1
            #         it.GoToNextItem()
            #
            #     nc = 0
            #     j = 0  # for field loop

            cc = vtk.vtkCellCenters()
            cc.SetInputData(reader.GetOutput())
            cc.Update()
            pts[:, :] = numpy_support.vtk_to_numpy(cc.GetOutput().GetPoints().GetData())

            for ifields in olist:
                    field = reader.GetOutput().GetCellData().GetArray(ifields)
                    nt = field.GetNumberOfValues()
                    f = numpy_support.vtk_to_numpy(field)
                    if ifields in self.name_indirection:
                        fielddict[self.name_indirection[ifields]] = f

            fielddict['sealtag'] = (fielddict['poro']==0.1)

        return pts, fielddict

    def _process_partial_(self, basename, suffix, mesh, i, max_region):
        """ 
        :param basename: base for the incremental name
        :param suffix: rest of the name
        :param mesh: block mesh as an unstructured mesh
        :param i: incrementer
        :param max_region: max increment to not overflow
        :return: 
        """
        if i < max_region:
            if mesh.GetArray(basename + str(i) + '_' + suffix) is not None:
                return mesh.GetArray(basename + str(i) + '_' + suffix)
            else:
                return self._process_partial_(basename, suffix, mesh, i + 1, max_region)
        else:
            raise LookupError("Field is missing")

    def process_keys(self, formula, fielddict):
        """
        Basic formula interface, reading from multiplication and division (no parenthesis)
        :param formula: string of the formula
        :param fielddict: all-containing dict with base-variables
        :return: Array of the formula value
        """
        import re
        if re.findall(r'if', formula):
            bits = formula.split(',')
            assert (len(bits) == 2)
            cdt = bits[0][3:]
            if re.findall(r'>', cdt):
                inner_bits = cdt.split('>')
                cdt_field = inner_bits[0].strip()
                cdt_lim = float(inner_bits[1])
                return np.where(fielddict[cdt_field] > cdt_lim, self.process_keys(bits[1][:-1], fielddict), 0)
            elif re.findall(r'<', cdt):
                inner_bits = cdt.split('<')
                cdt_field = inner_bits[0].strip()
                cdt_lim = float(inner_bits[1])
                return np.where(fielddict[cdt_field] < cdt_lim, self.process_keys(bits[1][:-1], fielddict), 0)
            else:
                raise NotImplemented('Not a valid condition')
        elif re.findall(r'\+', formula):
            bits = formula.split('+')
            bits = [item.strip() for item in bits]
            return self.process_keys(bits[0], fielddict) + self.process_keys('+'.join(bits[1:]), fielddict)
        elif re.findall(r'-', formula):
            bits = formula.split('-')
            bits = [item.strip() for item in bits]
            return self.process_keys(bits[0], fielddict) - self.process_keys('+'.join(bits[1:]), fielddict)
        elif re.findall(r'\*', formula):
            bits = formula.split('*')
            bits = [item.strip() for item in bits]
            return self.process_keys(bits[0], fielddict) * self.process_keys('*'.join(bits[1:]), fielddict)
        elif re.findall(r'/', formula):
            bits = formula.split('/')
            bits = [item.strip() for item in bits]
            return self.process_keys(bits[0], fielddict) / self.process_keys('/'.join(bits[1:]), fielddict)
        return fielddict[formula]
    def _process_solubility_(self, path):
        import pandas as pd
        df = pd.read_csv(path)
        p = df['pressure [Pa]'].to_numpy()
        T = np.arange(283, 345.5, 2.5)
        # note that solubility there is in molCo2/kgBrine
        S = df.to_numpy()[:, 1:].transpose()
        # convert to kgCO2/kgBrine
        S = S*Conversion.CO2MOL2KG
        p, T = np.meshgrid(p, T)
        return self._get_lambda_2_((p.flatten(), T.flatten()), S.flatten())
    def process(self, directory, ifile):
        """ god function to trigger processing """

        import os

        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")
        else:
            print(f"Directory already exists: {directory}")
