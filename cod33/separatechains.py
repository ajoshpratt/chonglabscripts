#!/usr/bin/env python
#
# separatechains.py
#
# Written 17.4.27 by Alex DeGrave
#
# Load 1rfo.pdb (the foldon trimer), and write each chain to a separate file.
#
#

import argparse
import MDAnalysis

class SeparateChains(object):
    def __init__(self):
        self._parse_args()

    def _parse_args(self):
        self.argparser = argparse.ArgumentParser()
        self.argparser.add_argument('--input', dest='inputpath', 
                                    default='1rfo_oriented.pdb',
                                    help="Load the foldon dimer from the "
                                         "specified PDB file."
                                    )
        self.args = self.argparser.parse_args()

    def go(self):
        self.u = MDAnalysis.Universe(self.args.inputpath) 
 
        self.protein = self.u.select_atoms('bynum 1-1299')
 
        self.chainA = self.u.select_atoms('bynum 1-432')
        self.chainB = self.u.select_atoms('bynum 433-864')
        self.chainC = self.u.select_atoms('bynum 865-1299')

        self.protein.translate(-1*self.protein.center_of_geometry())

        self.chainA.atoms.write('1rfo.A.pdb')
        self.chainB.atoms.write('1rfo.B.pdb')
        self.chainC.atoms.write('1rfo.C.pdb')
    

if __name__ == '__main__':
    sc = SeparateChains() 
    sc.go() 
