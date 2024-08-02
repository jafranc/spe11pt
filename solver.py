from abc import abstractmethod

from data import Data, Conversion
import re
import numpy as np
import pandas as pd


class Expressions:
    NEWTON_EXPRESSION = {'ndt': r'^New dt =(\s*[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?)',
                         'adt': r'accepted dt =(\s*[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?)',
                         'Iterations': r'Iterations: (\s*[1-9]\d*)',
                         'tsetup': r'Setup Time: ([0-9]+\.\d+) s',
                         'tsolve': r'Solve Time: ([0-9]+\.\d+) s'}


class Newton_iterations:

    def __init__(self):
        self.expression = Expressions.NEWTON_EXPRESSION
        self.collabels = list(self.expression.keys())
        self.collabels.insert(0, 'time')
        self.collabels.insert(1, 'dt')
        self.collabels.insert(2, 'nnl')
        self.collabels.insert(3, 'cut')

    def process(self, buffer, exp):
        solvers = []
        t = re.findall(exp, buffer)
        if len(t) > 0:
            for matched in t:
                solvers.append(float(matched))
        else:
            return (0, None)
        return len(t), np.sum(np.asarray(solvers), axis=0)
        # pass

    def extract_time(self, buffer):

        time = []
        ncut = 0
        for line in buffer.split('\n'):
            # m = re.search(self.expression['dt'], line)
            n = re.search(self.expression['ndt'], line)
            o = re.search(self.expression['adt'], line)
            if n:
                time.append(float(n.group(1)))
                ncut += 1
            elif o:
                time.append(float(o.group(1)))

        arr = np.asarray(time)

        return ncut,arr

    def _extract_newit_(self, buffer):
        #
        newit = []
        for line in buffer.split('\n'):
            m = re.search(r'NewtonIter:(\s*\d{1,2})', line)
            if m:
                newit.append(int(m.group(1)))

        arr = np.asarray(newit)
        pos = np.where(np.diff(arr) < 0)

        return arr[pos]


class Solver(Data):

    def __init__(self, simulator_name, version):
        super().__init__(simulator_name, version)
        self.NI = Newton_iterations()

    def process(self, directory, ifile):
        super().process(directory, ifile)

        buffer, time_tag, dt, header = self._split_log_(ifile)
        metadata = self._process_metadata_(header)
        values = pd.DataFrame(columns=self.NI.collabels)
        for i, bits in enumerate(buffer):
            new_row = {'time': time_tag[i + 1], 'dt': dt[i + 1]}

            self._process_bits_(bits, new_row)

            if len(new_row) == len(values.columns):
                values = pd.concat(
                    [pd.DataFrame(new_row, columns=values.columns, index=[i]), values.loc[:]]).reset_index(
                    drop=True)

        return values, metadata

    def _process_bits_(self, bits, new_row):
        for key, exp in self.NI.expression.items():
            nnl, value = self.NI.process(bits, exp)
            ncut,_ = self.NI.extract_time(bits)
            # TODO process accepted dt and new dt by re-spliting log
            if value is not None:
                if isinstance(value, int) or isinstance(value, float):
                    new_row[key] = value
                    new_row['nnl'] = nnl
                    new_row['cut'] = ncut
            else:
                new_row[key] = 0

    def _process_metadata_(self, header):
        nc = int(re.findall(r'C3D8: (\d+)', header)[0])
        return {'ncells': nc, 'ndof': 4 * nc}

    def _split_log_(self, fname):
        buffer = []
        time_tag = {}
        dt = {}
        buffer_ = ''
        with open(fname) as f:
            for line in f:
                if re.match(r'^Time', line):
                    buffer.append(buffer_)
                    time_tag[len(buffer)] = float(re.match(r'^Time: ((-)?\d+\.\d+e[+-]\d+)', line).group(1))
                    dt[len(buffer)] = float(re.findall(r'dt:(\s*(?:0|[1-9]\d*)(?:\.\d*)?)', line)[0])
                    buffer_ = ''
                elif re.match(r'Cleaning up events', line):
                    buffer.append(buffer_)
                else:
                    buffer_ += line
        header = buffer.pop(0)  # discard pre-simulation values
        return buffer, time_tag, dt, header
