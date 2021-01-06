'''

'''

import numpy as np
import networkx as nx

import csv

#### Auxillary Functions ####

def memristance(w, w_0=6e-9, R_on=100, R_off=10e4):
    ''''''
    
    M = R_on * w / w_0 + R_off * (1 - w / w_0)
    return M

def filament_change_1(I, w, u=1e-15, w_0=6e-9, R_on=100):
    '''
        Basic model for computing change in filament length in nanowire junction

        Parameters
        ----------
        I : float
            current current
        w : float 
            current filament length
        u : float
            ionic mobility
        w_0 : float
            junction seperation
        R_on : float
            on resistance

        Returns
        -------
        dwdt : float
            rate of change in filament length 
    '''

    dwdt = u * R_on * I / w_0
    
    return dwdt

def filament_change_2(I, w, u=1e-15, w_0=6e-9, R_on=100):
    '''
        Basic model for computing change in filament length in nanowire junction

        Parameters
        ----------
        I : float
            current current
        w : float 
            current filament length
        u : float
            ionic mobility
        w_0 : float
            junction seperation
        R_on : float
            on resistance

        Returns
        -------
        dwdt : float
            rate of change in filament length 
    '''

    omega = (w * (w_0 - w)) / w_0 * w_0

    dxdt = u * R_on * I * omega / (w_0*w_0)
    return dxdt

def heuns_step(func, x_0, y_0, h, *args):
    '''
    
    '''
    
    k_1 = func(x_0, y_0, *args)
    # k_2 = func(x_0 + h, y_0 + h*k_1, *args)
    
    y_1 = y_0 + (h/2)*(k_1 + k_1)
    
    return y_1

def memristor_step(I, w, R_on=100, w_0=6e-9, u_v=1e-15, dt=1e-5):
    ''''''

    # Arguments we need to pass to the filament growth model
    args = (u_v, w_0, R_on)

    # Take Heuns step
    w_next = heuns_step(filament_change_1, I, w, dt, *args)
    # Apply window functions
    if w_next > w_0:
        w_next = w_0
    elif w_next < 0:
        w_next = 0
    
    return w_next


#### Classes #### 

class RMNA:
    def __init__(self, M_model=memristance, M_step=memristor_step, M_args={}, \
        m_step_args={}):
        self.G = nx.Graph()
        self.w = dict()
        self.volt_In = dict()
        self.M_model = M_model
        self.M_args = M_args
        self.M_step = M_step
        self.m_step_args = m_step_args

        self.z = None
        self.A = None
        self.x = None
        self.x_sol = None

    def read_Graph(self, network):
        '''
            Reads a list of connections or nodes and creates a graph representation 
            from it.

            Paramters
            ---------
            network : list of lists
                List where each element of the list descriibes a connection 
                between nodes of the network. Each element should be of the form:
                    [(R|M|V), ID1, Val|None, ID2|None]
                The first element tells the type of the connection: R-resisitor,
                M-memristor, V-ID1 is a voltage input node. ID2 is not used if the 
                first element is V, but if Val is provided it will be used to set 
                the voltage injected. If the first element is R|M then Val should 
                either be the resistance of the connection or the wire gap. 

            Returns
            -------
            None

            Notes
            -----
            TODO: Extend function to allow current input nodes, must also extend 
            other functions to accomidate
        '''

        # Iterate over connections
        for connection in network:
            # If connection is a resistor
            if connection[0] == 'R':
                self.G.add_edge(connection[1], connection[2],\
                    weight=connection[3], element='R')
            # If connection is a memristor
            elif connection[0] == 'M':
                self.w[(connection[1], connection[2])] = connection[3]
                memristance = self.M_model(connection[3], **self.M_args)
                self.G.add_edge(connection[1], connection[2], weight=memristance,\
                     element='M')
            # If line is a voltage injector node
            elif connection[0] == 'V':
                self.G.add_node(connection[1])
                # Add node 
                if len(connection) >= 3:
                    self.volt_In[connection[1]] = connection[2]
                else:
                    self.volt_In[connection[1]] = 0

    def read_CSV(self, file_name, delimiter=','):
        '''
            Used to read connections directly from a csv file
        '''

        with open(file_name, mode='r') as file:
            csv_reader = csv.reader(file, delimiter=delimiter)
            for line in csv_reader:
                if line[0] == 'R':
                    self.G.add_edge(line[1], line[2], weight=line[3],\
                        element='R')

                # If connection is a memristor
                elif line[0] == 'M':
                    self.w[(line[1], line[2])] = line[3]
                    memristance = self.M_model(line[3], **self.M_args)
                    self.G.add_edge(line[1], line[2], weight=memristance,\
                        element='M')
                        
                # If line is a voltage injector node
                elif line[0] == 'V':
                    self.G.add_node(line[1])
                    # Add node 
                    if len(line) >= 3:
                        self.volt_In[line[1]] = line[2]
                    else:
                        self.volt_In[line[1]] = 0
        
    def update_Memristors(self, W):
        '''
            Updates the memristance stored in the graph structure. 

            Parameters
            ----------
            W : Iteratable of indexibles
                Expects to receive an iteratable object where each element is 
                indexible with two elements, the first being thememristor connection 
                represented as a tuple and the second its respective junction gaps.

            Notes
            -----
            TODO: Add error handling 
            TODO: Add ability to remove memristor connections
            TODO: Add ability to add new memristor connections
        '''
        
        for w in W:
            self.w[w[0]] = w[1]
            m = self.M_model(w[1], **self.M_args)
            self.G[w[0][0]][w[0][1]]['weight'] = m

    def update_Volt_In(self, volt_In):
        '''
            Updates the voltages stored in the graph structure. 

            Parameters
            ----------
            volt_In : Iteratable of indexibles
                Expects to receive an iteratable object where each element is
                indexible with two elements, the first being thevoltage node and
                the second its respective voltage.

            Notes
            -----
            TODO: Add error handling 
            TODO: Add ability to remove voltage nodes
            TODO: Add ability to inject voltage into new nodes
        '''
        
        for volt in volt_In:
            self.volt_In[volt[0]] = volt[1]

    def generate_MNA(self):
        '''
            Uses G representation of nanowire network to generate the A, x, and 
            z matrices used in modified nodal analysis of an electric network. 
            The x matrix is returned as a dictionary wherethe key is the name of 
            each node/plate and the value is which index of A and z it corresponds 
            to.
        '''

        #TODO: Add support for current injection
        #TODO: Optimize, possible by caching some info about matrix structure

        # Error checking
        if self.G is None:
            print('Error: No G representation, use graph_jda or graph_mnr first.')
            return None

        # Initilize
        self.A = np.zeros((len(self.G) + len(self.volt_In), len(self.G) + len(self.volt_In)))
        self.x = {}
        self.z = np.zeros(len(self.G) + len(self.volt_In))

        # Loop over nodes(v) in the G
        for v in self.G:
            # Get/give index for/to node
            n = self.x.setdefault(str(v), len(self.x))

            ## Construct resistance (G) matrix ##
            total = 0
            for w in self.G.adj[v]: # Look at each adjacent node 
                # Get/give index for/to adjacent node
                m = self.x.setdefault(str(w), len(self.x)) 

                # Compute diagonal and off diagonals of G
                total += 1 / self.G.edges[v, w]['weight']
                self.A[n][m] = - 1 / self.G.edges[v, w]['weight']
                
            # Set diagonal of G
            self.A[n][n] = total

            ## Construct voltage/current (C/B/z) of system ##
            if v in self.volt_In:
                # Get/give index for/to adjacent node
                m = self.x.setdefault('i' + str(v), len(self.x))

                # Add entry to C/B/z
                self.A[m][n] = 1
                self.A[n][m] = 1
                self.z[m] = self.volt_In[v]
   

        ## Remove zero rows and columns ##
        self.A = self.A[~(self.A==0).all(1)]
        self.A = self.A[:, ~(self.A==0).all(0)]
        self.z = self.z[:len(self.A)]

        return self.A, self.x, self.z

    def solve_MNA(self):
        ''''''

        # TODO: Make the method of solving flexible
        self.x_sol = np.linalg.solve(self.A, self.z)
        return self.x_sol

    def get_Current(self, junctions):
        ''''''

        if self.x_sol is None or self.x is None:
            print("ERROR: Must generate and solve MNA system first")
            return

        # TODO: Make this worrk for all junctions not just memrstors
        currents = dict()
        for j in junctions:
            if j in self.w:
                # Get voltage values across junction
                idx_a = self.x[str(j[0])]
                V_a = self.x_sol[idx_a]
                idx_b = self.x[str(j[1])]
                V_b = self.x_sol[idx_b]

                # Compute current
                # NOTE:  Must be considerate of the direction of voltage/current
                V = V_b - V_a
                M = self.M_model(self.w[j], **self.M_args)
                I = M * V
                currents[j] = I

            else:
                print("ERROR: No such juncton exists.")
                return

        return currents

    def run_RMNA(self, volt_In_Series):
        ''''''

        # Get list of junctions
        junctions = list(self.w.keys())

        # Create lists for storing the V,I,M for each junction
        # TODO: Lists are inefficient, turn these into Pandas dataframes
        node_V_series = [junctions]
        node_I_series = [junctions]
        M_series = [junctions]

        # Run time series
        for n, volts in enumerate(volt_In_Series):
            self.update_Volt_In(volts)

            # Create and solve MNA system
            self.generate_MNA()
            self.solve_MNA()

            # Update voltage series for memristors


            # Compute currrents
            Is = self.get_Current(junctions)
            node_I_series.append(Is)

            # Update Memristance
            # TODO: Maybe optimize this so the function only calls once
            M_arr = []
            for m, j in enumerate(junctions):
                M = self.M_step(Is[m], self.w[j])
                M_arr.append(M)
                self.update_Memristors([(j, M)])

            M_series.append(M_arr)

        return node_I_series, M_series, node_V_series
