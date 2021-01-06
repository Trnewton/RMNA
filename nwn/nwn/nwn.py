'''

'''

import numpy as np
import networkx as nx

from shapely.geometry import LineString
from shapely.geometry import MultiLineString
import matplotlib.pyplot as plt

import heapq

class nanowire_network:
    '''
    

    Attributes
    ----------
    wires : shapely MultiLineString
        collection of shapely linestring objects whcih represent the physical nanowire network
    plates : list
        a list of tuples of the form (Linestring, "[N,I,V]", float) where the linestring object 
        gives the physical representation of the plate, the string where the plate is current 
        or voltage injecting, or neither, and the float the potential or current on the plate
    intercepts : list
        list of tuples of the form ((x, y), "W[0-9]+", "[W,P][0-9]+") where the first tuple (x, y) 
        details the location of the intercept, and the strings the two wires or plate which intercept 
    graph : networkx Graph
        graphical representation of network, formed using either the Junction Dominated Assumption
        or the Multi-Nodular Representation
    graph_representation : str
        string detailing which graphical model was used to generate the graph attribute

    Methods
    -------
    set_plates : list 
        

    '''

    def __init__(self, num_wires, wire_length, width, height, creation_method='midpoint', seed=None ):
        '''
            Parameters
            ----------
            num_wires : Number of wires to randomly generate 

            wire_length : Length of each wire

            width : Width of box wires will be generated in 

            height : Height of box wires will be generated in 

            creation_method : Method for generating location and orientation of each wire, 
                              one of 'midpoint' or 'endpoint' (default='midpoint')

            seed : Random number genrator seed
        '''

        if creation_method not in ('midpoint', 'endpoint'):
            raise ValueError('unsupported creation method: {}'.format(creation_method))
        elif creation_method == 'endpoint':
            self.wires = MultiLineString(create_nwn_end(num_wires, width, height, wire_length, seed))
        elif creation_method == 'midpoint':
            self.wires = MultiLineString(create_nwn_mid(num_wires, width, height, wire_length, seed))

        self.plates = None
        self.intercepts = None
        self.graph = None
        self.graph_representation = None

    def set_plates(self, plates):
        '''
            Creates plates with with initial potential or current.
            plates: [((x1, y1), (x2, y2), "I/V/N", value), ... ]
        '''

        # 
        self.plates = []
        self.plat_vals = []
        if plates is not None:
            for plate in plates:
                self.plates.append(LineString(plate[0:2]))
                self.plat_vals.append(plate[2:4])

    def charge_plates(self, new_plates):
        '''
            INCOMPLETE. Changes plates potential or applied current.
        '''
        
        # TODO: Create method to allow for already established plates to have new current/voltage values

    def calc_intercepts(self):
        '''
        Calculates the intercepts for the nanowire network 
        '''

        self.intercepts = []

        # Find intercepts between wires
        for n in range(len(self.wires)):
            for m in range(n, len(self.wires)):
                if self.wires[n].crosses(self.wires[m]):
                    p = self.wires[n].intersection(self.wires[m])
                    self.intercepts.append(((p.x, p.y), 'W'+str(n), 'W'+str(m)))

        # Find intercepts between wires and plates
        for n, plate in enumerate(self.plates):
            for m, wire in enumerate(self.wires):
                if wire.crosses(plate):
                    p = wire.intersection(plate)
                    self.intercepts.append(((p.x, p.y),'P'+str(n), 'W'+str(m)))
        
        # Assuming plates do not short we do not need check if they intersect

        return self.intercepts

    def draw(self):
        '''
        Draws nanowire network with intercepts if they have been calculated
        '''

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # Plot wires with labels
        for n, wire in enumerate(self.wires):
            x, y = wire.xy
            ax.plot(x, y, color='c')
            ax.plot(x[0], y[0], 'o', ms=4, color='b')
            ax.plot(x[1], y[1], 'o', ms=4, color='g')
            ax.text((x[0] + x[1])/2, (y[0] + y[1])/2, 'W' + str(n))

        # Plot plates with labels
        for n, plate in enumerate(self.plates):
            x, y = plate.xy
            ax.plot(x, y, color='y')
            ax.text(x[1], y[1], 'P' + str(n))

        # Plot intercepts
        if self.intercepts is not None:
            for intercept in self.intercepts:
                x, y = intercept[0]
                ax.plot(x, y, '*' , ms=6, color='k')     

        plt.show()

        return fig, ax

    def graph_jda(self, junction_res=1): 
        '''
        Creates graph of nanowire network using the Junction Dominated Assumption, returning a networkx 
        graph object with nodes either labelled with a 'P' to indicate a plate or a 'W' to indicate a 
        wire followed by the order in which the plate/wire was labelled. 
        '''
        
        self.graph_representation = 'Junction Dominated Assumption'
        self.graph = nx.Graph()

        if self.intercepts is None:
            self.calc_intercepts()

        # Use precalculated intercepts to create graph
        for __, wire1, wire2 in self.intercepts:
            self.graph.add_edge(wire1, wire2, weight=junction_res)

        return self.graph

    def graph_mnr(self, resistance_density=1, memristor=1):
        '''
        Generates graph for multi-nodular representation of the nanowire network.
        '''
         
        self.graph_representation = 'Multi-Nodal Representation'
        self.graph = nx.Graph()

        wire_dict = {}

        for pos, wire1, wire2 in self.intercepts:
            heap1 = wire_dict.setdefault(wire1, [])
            if 'P' not in wire1:
                int1 = (pos, wire1 + 'J' + str(len(heap1)) )
            else:
                int1 = (pos, wire1)

            heapq.heappush(heap1, int1)

            heap2 = wire_dict.setdefault(wire2, [])
            int2 = (pos, wire2 + 'J' + str(len(heap2)) )
            heapq.heappush(heap2, int2)

            self.graph.add_edge(int1[1], int2[1], weight=memristor) 

        for wire in wire_dict:
            if 'P' not in wire:
                heap = wire_dict.get(wire, [])
                if heap:
                    int1 = heapq.heappop(heap)
                while heap:
                    int2 = int1
                    int1 = heapq.heappop(heap)

                    int_dist = np.sqrt( (int1[0][0] - int2[0][0])**2 + (int1[0][1] - int2[0][1])**2 )
                    self.graph.add_edge(int1[1], int2[1], weight=resistance_density*int_dist)

        return self.graph
  
    def mna_matrix(self):
        '''
            Uses graph representation of nanowire network to generate the A, x, and z matrices used in 
            modified nodal analysis of an electric network. The x matrix is returned as a dictionary where
            the key is the name of each node/plate and the value is which index of A and z it corresponds to.
        '''

        if self.graph is None:
            print('Error: No graph representation, use graph_jda or graph_mnr first.')
            return None

        A = np.zeros((len(self.graph) + len(self.plates), len(self.graph) + len(self.plates)))
        x = {}
        z = np.zeros(len(self.graph) + len(self.plates))

        for v in self.graph:
            n = x.setdefault(v, len(x))

            # Construct resistance (G) matrix
            total = 0
            for w in self.graph.adj[v]:
                m = x.setdefault(w, len(x))
                total += 1 / self.graph.edges[v, w]['weight']
                A[n][m] = - 1 / self.graph.edges[v, w]['weight']
            A[n][n] = total

            # Construct voltage/current (C/B) of matrix
            if 'P' in v:
                if self.plat_vals[int(v[1])][0] == 'V':
                    m = x.setdefault('i' + v, len(x))
                    A[m][n] = 1
                    A[n][m] = 1
                    z[m] = self.plat_vals[int(v[1])][1]
                elif self.plat_vals[int(v[1])][0] == 'I':
                    z[n] = self.plat_vals[int(v[1])][1]

        # Remove zero rows and columns
        A = A[~(A==0).all(1)]
        A = A[:, ~(A==0).all(0)]
        z = z[:len(A)]

        return A, x, z


class memristor:
    '''

    '''

    def __init__(self, res_on, res_off, wire_gap, ionic_mobil, model, step_method, filament_length=0):
        '''
        
        '''

        self.res_on = res_on
        self.res_off = res_off
        self.wire_gap = wire_gap
        self.ionic_mobil = ionic_mobil
        self.model = model
        self.step_method = step_method
        self.filament_length = filament_length

    def time_step(self, dt, I):
        '''
        #TODO : Calculates new resistance after a given time step based on current state
        '''
        
       # dw = self.model(I, self.ionic_mobil, self.wire_gap, self.res_on, I, self.filament_length)

        kawrds = {'I':I, 'u':self.ionic_mobil, 'w_0':self.wire_gap, 'R_o':self.res_on}
        self.filament_length = self.step_method(self.model,  **kawrds)
 
    def resistance(self):
        '''
        '''
        #TODO : 
        res = self.res_on * (self.filament_length/self.wire_gap) + self.res_off * (1 - self.filament_length/self.wire_gap)

        return res
        

## Functions ## 

def create_nwn_end(N, X, Y, l0, seed=None):
    '''
    Generates a list of endpoints for a random network of wires using the "end-point" method
    '''

    import numpy as np

    if seed is not None:
        np.random.seed(seed)


    l = l0#Make random
    pos1 = np.concatenate((np.random.rand(N,1)*X, np.random.rand(N,1)*Y), axis=1)
    theta = np.random.rand(N)*2*np.pi
    pos2 = pos1 + (np.array([np.cos(theta), np.sin(theta)])*l).T

    return list(zip(pos1, pos2))

def create_nwn_mid(N, X, Y, l0, seed=None):
    '''
    Generates a list of endpoints for a random network of wires using the "mid-point" method
    '''

    import numpy as np
     
    if seed is not None:
        np.random.seed(seed)

    l = l0#Make random
    center = np.array([np.random.rand(N)*X, np.random.rand(N)*Y]).T
    theta = np.random.rand(N)*2*np.pi
    dp = (np.array([np.cos(theta), np.sin(theta)])*l/2).T
    pos1 = center + dp
    pos2 = center - dp

    return list(zip(pos1, pos2))

def filament_change_1(w, I, u, w_0, R_o):
    '''
    Basic model for computing change in filament length in nanowire junction

    Parameters
    ----------
    u : float
        ionic mobility
    w_0 : float
        junction seperation
    R_o : float
        on resistance
    I : float
        current current
    w : float 
        current filament length

    Returns
    -------
    dwdt : float
        rate of change in filament length 

    '''

    #TODO : Optimze
    omega = (w * (w_0 - w)) / w_0 * w_0
    return u * R_o * I * omega / w_0

def eulers_method_step(func, x, dt):
    return x + func(x)*dt


def HeunsStep(func, x0, y0, h):
    '''

    '''

    # Calculate intermediate steps
    k1 = func(x0, y0)
    k2 = func(x0 + h, y0 + k1*h)

    # Calculate next y
    yf = y0 + 0.5*h*(k1 + k2)

    return yf
