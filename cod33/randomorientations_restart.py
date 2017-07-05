#!/usr/bin/env python
#
# random_orientation.py
#
# Randomly orient 3 foldon trimers such that the centers of mass for each
# subunit forms an equilateral triangle with a specified edge length. 
#
# Written 17.4.27 by Alex DeGrave
#
from __future__ import print_function
import argparse
import MDAnalysis
import MDAnalysis.analysis.distances
import numpy
import sys

class RandomOrientation(object):
    def __init__(self):
        self.parse_args()
        self.edgelength = self.args.edgelength 
        self.load_chains()
        self.threshold=0.01

    def parse_args(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--edgelength', dest='edgelength', default=40,
                                 type=float,
                                 help="Separate foldon subunits by the " 
                                      "specified length, in Angstroms" 
                                 )

        self.parser.add_argument('--inputpath', dest='inputpath', 
                                 default='restart.file.initial',
                                 help="Path to the restart file for the foldon "
                                      "trimer (should be something like "
                                      "restart.file.initial)."
                                 )

        self.parser.add_argument('--outputpath', dest='outputpath', 
                                 default='restart.file.initial.oriented',
                                 help="Path to which a new irestart file will "
                                      "be output, containing the 3 monomers at"
                                      " the requested separation and rotated "
                                      "randomly."
                                 )
     
        self.parser.add_argument('--surface', dest='surface', 
                                 action='store_true',
                                 help="If this argument is supplied, calculate "
                                      "positions of the monomers such that the "
                                      "nearest atoms in each monomer are "
                                      "'edgelength' Angstroms apart.  If this "
                                      "argument is not supplied, base the "
                                      "calculation on the center of mass."
                                 )

        

        self.args = self.parser.parse_args()

    def random_rotation_matrix(self):
        '''
        Generate a random (uniform) 3 by 3 rotation matrix.  
        '''
        # Pick a random point on the unit sphere (S3)
        # See http://mathworld.wolfram.com/SpherePointPicking.html
        theta = numpy.random.random()*2*numpy.pi
        phi = numpy.arccos(2*numpy.random.random()-1)
        x = numpy.cos(theta)*numpy.sin(phi)
        y = numpy.sin(theta)*numpy.sin(phi)
        z = numpy.cos(phi)

        # The uniform random rotation matrix is then given by the rotation
        # matrix U such that Uv1=v2, as shown below.
        v1 = numpy.array((1,0,0))
        v2 = numpy.array((x,y,z))

        # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        v_cross = numpy.cross(v1, v2)        
        m = numpy.array(((0            ,-1*v_cross[2],    v_cross[1]),
                         (   v_cross[2], 0           , -1*v_cross[0]),
                         (-1*v_cross[1],   v_cross[0], 0            )))

        U = numpy.identity(3) + m + m.dot(m)*(1/(1+v1.dot(v2)))

        return U

    def load_chains(self):
        '''
        Load the restart.file and make it into an MDAnalysis Universe. Then 
        set self.chainA ... self.chainC to MDAnalysis selections corresponding
        to each chain.
        '''
        initial_coords = numpy.loadtxt(self.args.inputpath, skiprows=1, 
                                       usecols=(2,3,4))
        self.junkA = initial_coords[0,:] 
        self.junkB = initial_coords[71,:] 
        self.junkC = initial_coords[142,:] 

        initial_coords = numpy.concatenate((initial_coords[1:71], 
                                            initial_coords[72:142], 
                                            initial_coords[143:213]))

        self.u = MDAnalysis.Universe('example.pdb', 'example.pdb')
        self.u.atoms.positions = initial_coords
        self.chainA = self.u.select_atoms('bynum 1-70')
        self.chainB = self.u.select_atoms('bynum 71-140')
        self.chainC = self.u.select_atoms('bynum 141-210')

    def translate_equilateral(self, edgelength):

        pos1 = numpy.array((self.edgelength/numpy.sqrt(3),
                            0,
                            0)
                           )

        pos2 = numpy.array((-1./2*self.edgelength/numpy.sqrt(3),
                            self.edgelength/2,
                            0)
                           )
 
        pos3 = numpy.array((-1./2*self.edgelength/numpy.sqrt(3),
                            -1*self.edgelength/2,
                            0)
                           )

        self.chainA.translate(pos1)
        self.chainB.translate(pos2)
        self.chainC.translate(pos3)

    def separate_surfaces(self):
        distarray = MDAnalysis.analysis.distances.distance_array
        # Calculate the protein diameter
        diams = []
        for chain in [self.chainA, self.chainB, self.chainC]:
            dists = distarray(chain.positions,
                              chain.positions)
            diams.append(dists.max())
        # Take a guess at the correct separation
        
        self.translate_equilateral(max(diams)+self.edgelength)

        error = self.threshold*10 
        # Iteratively move the subunits to achieve the desired separation
        # between the surfaces.
        div = 2
        while error > self.threshold:

            # Calculate the vectors between the centers of geometry, so
            # that we know which way to move.
            p1 = self.chainA.center_of_geometry()
            p2 = self.chainB.center_of_geometry()
            v12 = p1-p2
            v12 /= numpy.linalg.norm(v12)
       
            # Start by moving chain A          
            distAB = distarray(self.chainA.positions,
                               self.chainB.positions).min()
            self.chainA.translate(v12*(self.edgelength-distAB)/div)


            # Calculate the vectors between the centers of geometry, so
            # that we know which way to move.
            p2 = self.chainB.center_of_geometry()
            p3 = self.chainC.center_of_geometry()
            v23 = p2-p3
            v23 /= numpy.linalg.norm(v23)
       
            # Move chain B 
            distBC = distarray(self.chainB.positions,
                               self.chainC.positions).min()
            self.chainB.translate(v23*(self.edgelength-distBC)/div)


            # Calculate the vectors between the centers of geometry, so
            # that we know which way to move.
            p1 = self.chainA.center_of_geometry()
            p3 = self.chainC.center_of_geometry()
            v31 = p3-p1
            v31 /= numpy.linalg.norm(v31)
       
            # Move chain C 
            distAC = distarray(self.chainA.positions,
                               self.chainC.positions).min()
            self.chainC.translate(v31*(self.edgelength-distAC)/div)

            error  = abs(distarray(self.chainA.positions,
                                   self.chainB.positions).min() \
                         - self.edgelength)
            error += abs(distarray(self.chainB.positions,
                                   self.chainC.positions).min() \
                         - self.edgelength)
            error += abs(distarray(self.chainA.positions,
                                   self.chainC.positions).min() \
                         - self.edgelength)

            sys.stdout.write("\rSummed deviation from ideal separation: "
                             "{:.04f}".format(error))
            sys.stdout.flush()

    def _format_line(self, chain, resid, pos_arr):
        '''
        Inputs:
          - chain: the chain index (int)
          - resid: the residue index (int)
          - pos_arr: length 3 array of x,y,z coords (float)
        Returns:
          - a string corresponding to one line in the restart.file
        '''
        x = pos_arr[0]
        y = pos_arr[1]
        z = pos_arr[2]

        l = "{:d}".format(chain).rjust(8) +\
            "{:d}".format(resid).rjust(8) +\
            "{:.5f}".format(x).rjust(20) +\
            "{:.5f}".format(y).rjust(20) +\
            "{:.5f}".format(z).rjust(20) +\
            "\n"
        return l

    def _write(self):
        lines = []
        time_line = 'time =' + '0.00000'.rjust(21) + ' ps\n'
        lines.append(time_line)

        # Chain A
        # The weird line that has resid 0
        lines.append(self._format_line(1,0, self.junkA))
        for i in range(0,len(self.chainA.atoms.positions)):
            resid = i+1 # change from zero indexing to one-indexing
            lines.append(self._format_line(1, 
                                           resid,
                                           self.chainA.positions[i]
                                           )                   
                         )

        # Chain B
        lines.append(self._format_line(2,0, self.junkB))
        for i in range(0,len(self.chainB.atoms.positions)):
            resid = i+1 # change from zero indexing to one-indexing
            lines.append(self._format_line(2, 
                                           resid,
                                           self.chainB.positions[i]
                                           )                   
                         )

        # Chain C
        lines.append(self._format_line(3,0, self.junkC))
        for i in range(0,len(self.chainC.atoms.positions)):
            resid = i+1 # change from zero indexing to one-indexing
            lines.append(self._format_line(3, 
                                           resid,
                                           self.chainC.positions[i]
                                           )                   
                         )

        with open('restart.file.initial.oriented', 'w+') as outfile:
            for line in lines:
                outfile.write(line)

    def go(self) :
        '''
        The main method.  Run everything.
        '''

        for chain in [self.chainA, self.chainB, self.chainC]:
            chain.atoms.translate(-1*chain.center_of_geometry())

        for chain in [self.chainA, self.chainB, self.chainC]:
            rand_rot_mat = self.random_rotation_matrix()
            chain.rotate(rand_rot_mat)

        if not self.args.surface:
            self.translate_equilateral(self.edgelength)
        else:
            self.separate_surfaces()

        self._write()


if __name__ == "__main__":
    r = RandomOrientation() 
    r.go()
