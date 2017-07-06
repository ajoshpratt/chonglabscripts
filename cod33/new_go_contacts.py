#!/usr/bin/env python
#Corinn and Alex 5 Ever science 17.5.19
import numpy

class GoCorrector(object):
    def __init__(self):
        self.input_A_path = 'A.go.parameters'
        self.input_B_path = 'B.go.parameters'
        self.input_C_path = 'C.go.parameters'
        self.input_AB_path = 'AB.go.parameters'
        self.input_AC_path = 'AC.go.parameters'
        self.input_BC_path = 'BC.go.parameters'

        self.all_paths = [self.input_A_path,
                          self.input_B_path,
                          self.input_C_path,
                          self.input_AB_path,
                          self.input_AC_path,
                          self.input_BC_path]
        self._parse()
        self._combine()
        self._write()

    def _parse(self):
        '''
        Parses the go parameters files specified by:
            self.input_A_path 
            self.input_B_path 
            self.input_C_path 
            self.input_AB_path
            self.input_BC_path
            self.input_AC_path
        and creates attributes 
            self.input_A
            ...
            self.input_AC
        each of which are dictionaires of tuples
           (resid1, resid2): (sigma, epsilon)
        '''
        self.input_A = {}
        self.input_B = {}
        self.input_C = {}
        self.input_AB = {}
        self.input_AC = {}
        self.input_BC = {}
        self.all_inputs = [self.input_A,
                           self.input_B,
                           self.input_C,
                           self.input_AB,
                           self.input_AC,
                           self.input_BC]

        for ipath, path in enumerate(self.all_paths):
            with open(path, 'r') as f:
                f.readline()
                for line in f:
                    s = line.split()
                    resid1 = int(s[1])
                    resid2 = int(s[3])
                    sigma  = float(s[4])
                    epsilon = float(s[5])
                    self.all_inputs[ipath][(resid1, resid2)] = (sigma, epsilon)
                            

    def _combine(self):
        '''
        Combines the parsed go parameters in 
            self.input_A
            ...
            self.input_AC
        and creates new attributes
            self.intra_contacts
            self.inter_contacts
        each of which are lists of tuples
            (resid1, resid2, sigma, epsilon)
        '''
        keysA = self.input_A.keys()
        keysB = self.input_B.keys()
        keysC = self.input_C.keys()
        # keysA = [(1,2), (1,5), ... (67,70)]
        intra_keys = list(set(keysA) | set(keysB) | set(keysC))
        self.intra_contacts = []
        for intra_key in intra_keys:
            # Find the mean sigma, if the key exists in multiple input files
            sigmas = []
            epsilon = None
            for input_file in [self.input_A, self.input_B, self.input_C]:
                try: 
                    sigmas.append(input_file[intra_key][0])
                    epsilon = input_file[intra_key][1]
                except KeyError:
                    pass
            if len(sigmas) == 0 or epsilon is None:
                raise ValueError
            sigma = numpy.array(sigmas).min()
            self.intra_contacts.append(
                    (intra_key[0], intra_key[1], sigma, epsilon)
                                       )

        keysAB = self.input_AB.keys()
        keysAC = self.input_AC.keys()
        keysBC = self.input_BC.keys()
        inter_keys = list(set(keysAB) | set(keysAC) | set(keysBC))

        self.inter_contacts = []
        for inter_key in inter_keys:
            # Find the mean sigma, if the key exists in multiple input files
            sigmas = []
            epsilon = None
            for input_file in [self.input_AB, self.input_AC, self.input_BC]:
                try: 
                    sigmas.append(input_file[inter_key][0])
                    epsilon = input_file[inter_key][1]
                except KeyError:
                    pass
            if len(sigmas) == 0 or epsilon is None:
                raise ValueError
            sigma = numpy.array(sigmas).min()
            self.inter_contacts.append(
                    (inter_key[0], inter_key[1], sigma, epsilon)
                                       )

    def _format_line(self, chain1, resid1, chain2, resid2, sigma, epsilon):
        l = "{:d}".format(chain1).rjust(10) +\
            "{:d}".format(resid1).rjust(10) +\
            "{:d}".format(chain2).rjust(10) +\
            "{:d}".format(resid2).rjust(10) +\
            "{:.5f}".format(sigma).rjust(15) +\
            "{:.5f}".format(epsilon).rjust(15)
        return l

    def _write(self):
        '''
        Writes out new go.parameters files, using the parameters created by
        self._combine. Parameters are written to:
            'A.go.parameters.new'
            ...
            'AC.go.parameters.new'
        '''
        lines = []
        for intra_contact in self.intra_contacts:
            for chain in [1,2,3]:
                lines.append(self._format_line(
                        chain,
                        intra_contact[0],
                        chain,
                        intra_contact[1],
                        intra_contact[2],
                        intra_contact[3]
                ))
        for inter_contact in self.inter_contacts:
            for chain_1 in [1, 2, 3]:
                for chain_2 in [1, 2, 3]:
                    if chain_1 != chain_2:
                        lines.append(self._format_line(
                            chain_1,
                            inter_contact[0],
                            chain_2,
                            inter_contact[1],
                            inter_contact[2],
                            inter_contact[3]
                        ))
        lines.sort()
        lines.insert(0,'12-10 potentials')
        with open('system.go.parameters', 'w+') as outfile:
            for line in lines:
                outfile.write(line+'\n')

if __name__ == '__main__':
    gc = GoCorrector()
