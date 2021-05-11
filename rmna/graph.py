'''

'''

# Base Python Imports
import csv, random, heapq

# Third Part Imports
import numpy as np
import networkx as nx

from shapely.geometry import LineString
from shapely.geometry import MultiLineString



########################### Random Nanowire Networks ###########################

def create_nwn_end(N, X=1, Y=1, l0=1, seed=None):
    '''
    Generates a list of endpoints for a random network of wires using the "end-point" method
    '''

    if seed is not None:
        np.random.seed(seed)

    l = l0 #TODO: Make random variable with given pdf

    pos1 = np.concatenate((np.random.rand(N,1)*X, np.random.rand(N,1)*Y), axis=1)
    theta = np.random.rand(N)*2*np.pi
    pos2 = pos1 + (np.array([np.cos(theta), np.sin(theta)])*l).T

    return MultiLineString(list(zip(pos1, pos2)))

def create_nwn_mid(N, X=1, Y=1, l0=1, seed=None):
    '''
        Generates a list of endpoints for a random network of wires using the 
        "mid-point" method
    '''
     
    if seed is not None:
        np.random.seed(seed)

    l = l0 #TODO: Make random variable with given pdf
    
    center = np.array([np.random.rand(N)*X, np.random.rand(N)*Y]).T
    theta = np.random.rand(N)*2*np.pi
    dp = (np.array([np.cos(theta), np.sin(theta)])*l/2).T
    pos1 = center + dp
    pos2 = center - dp

    return MultiLineString(list(zip(pos1, pos2)))

def calc_intercepts(wires, plates=None):
    ''''''

    intercepts = []

        # Find intercepts between wires
    for n in range(len(wires)):
        for m in range(n, len(wires)):
            if wires[n].crosses(wires[m]):
                p = wires[n].intersection(wires[m])
                intercepts.append(((p.x, p.y), 'W'+str(n), 'W'+str(m)))

    # Find intercepts between wires and plates
    for n, plate in enumerate(plates):
        for m, wire in enumerate(wires):
            if wire.crosses(plate):
                p = wire.intersection(plate)
                intercepts.append(((p.x, p.y),'P'+str(n), 'W'+str(m)))
    
        # Assuming plates do not short we do not need check if they intersect

    return intercepts

def JDA_list(intercepts, w=6e-9):
    ''''''

    edges = []

    for __, wire1, wire2 in intercepts:
        edges.append(('M', wire1, wire2, w))

    return edges

def JDA_csv(intercepts, junction_res=1):
    raise NotImplementedError
    
def MNR_list(intercepts, w=6e-9, rho=1):
    ''''''

    edges = []

    wire_dict = {}

    for pos, wire1, wire2 in intercepts:
        heap1 = wire_dict.setdefault(wire1, [])
        if 'P' not in wire1:
            int1 = (pos, wire1 + 'J' + str(len(heap1)) )
        else:
            int1 = (pos, wire1)

        heapq.heappush(heap1, int1)

        heap2 = wire_dict.setdefault(wire2, [])
        int2 = (pos, wire2 + 'J' + str(len(heap2)) )
        heapq.heappush(heap2, int2)

        edges.append(('M', int1[1], int2[1], w))

    for wire in wire_dict:
        if 'P' not in wire:
            heap = wire_dict.get(wire, [])
            if heap:
                int1 = heapq.heappop(heap)
            while heap:
                int2 = int1
                int1 = heapq.heappop(heap)

                int_dist = np.sqrt( (int1[0][0] - int2[0][0])**2 + \
                                    (int1[0][1] - int2[0][1])**2 )

                edges.append(('R', int1[1], int2[1], rho*int_dist))

    return edges

def MNR_csv():
    raise NotImplementedError

############################ Grid Nanowire Networks ############################

def list_Grid(N, H, w):
    '''
        Creates a NxH grid of memristors as a list representation
    '''
    
    grid = []
    
    for n in range(N-1):
        for h in range(H-1):
            grid.append(['M', n*H+h, n*H+h+1, w])
            grid.append(['M', n*H+h, n*H+h+H, w])
            
    for n in range(N-1):
        grid.append(['M', (n+1)*H-1, (n+2)*H-1, w])
        
    for h in range(H-1):
        grid.append(['M', (N-1)*H+h, (N-1)*H+h+1, w])
            
    return grid
    

############################ Maze Nanowire Networks ############################

def _grid_dict(N, H):
    '''
    '''
    
    grid = dict()
    
    for n in range(N-1):
        for h in range(H-1):
            grid.setdefault(n*H+h, []).append(n*H+h+1)
            grid.setdefault(n*H+h+1, []).append(n*H+h)
            grid.setdefault(n*H+h, []).append(n*H+h+H)
            grid.setdefault(n*H+h+H, []).append(n*H+h)
            
    for n in range(N-1):
        grid.setdefault((n+1)*H-1, []).append((n+2)*H-1)
        grid.setdefault((n+2)*H-1, []).append((n+1)*H-1)
        
    for h in range(H-1):
        grid.setdefault((N-1)*H+h, []).append((N-1)*H+h+1)
        grid.setdefault((N-1)*H+h+1, []).append((N-1)*H+h)
            
    return grid

def wilson_maze(N, H, start, finish):
    '''
    '''
    
    maze = {finish:[]}
    grid = _grid_dict(N, H)
    
    to_add = list(grid.keys())
    to_add.remove(finish)
    
    path = dict()
    
    while to_add:
        cur_node = start
        while cur_node not in maze:
            neighbours = grid[cur_node]
            step = random.choice(neighbours)
            path[cur_node] = step

            if step in maze:
                node = start
                while node not in maze:
                    to_add.remove(node)
                    maze[node] = [path[node]]
                    node = path[node]
                    
                path = dict()
                if to_add:
                    start = random.choice(to_add)

            else:
                cur_node = step
            
    return maze
    
def single_path_2_mulit(maze, num_new_paths):
    '''
    '''
    
    pass



if __name__ == "__main__":
    N = 4
    H = 4
    start = 0
    finish = 15
    print(wilson_maze(N, H, start, finish))