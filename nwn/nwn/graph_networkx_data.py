import networkx as nx
import heapq 


#### Path finding ####
def all_shortest_path_unweighted_single(G, start):
    '''
    Finds all the shortest paths from a start node to each node of a graph
    '''

    adj = G.adj
    visited = {start:[start,]}
    new_nodes = [start]
    add = new_nodes.append
    while new_nodes:
        cur_nodes = new_nodes
        new_nodes = []
        for node in cur_nodes:
            for neighbour in adj[node]:
                if neighbour not in visited:
                    visited[neighbour] = visited[node] + [neighbour,]
                    add(neighbour)
    return visited

def all_shortest_path_unweighted(G):
    '''
    Finds all the shortest paths in a graph
    '''

    # It is possible to make this more efficient by using previous searches
    # for example in an undirected graph the last node looked at, all the shortest
    # paths should already be known

    paths = {}
    for node in G:
        paths.update(all_shortest_path_unweighted_single(G, node))

    return paths

def shortest_path_unweighted(G, start, end):    
    adj = G.adj
    start_visited = {start:[start,]}
    end_visited = {end:[end,]}
    start_edge = [start]
    add_start = start_edge.append
    end_edge = [end]
    add_end = end_edge.append

    while start_edge or end_edge:
        if len(start_edge) > len(end_edge):
            cur_node = end_edge.copy()
            end_edge.clear()
            for node in cur_node:
                for nbr in adj[node]:
                    if nbr in start_visited:
                        ### Found path
                        # Combined routes and return
                        path = start_visited[nbr] + end_visited[node][::-1]
                        return path
                    elif nbr not in end_visited:
                        # Extend end ring
                        end_visited[nbr] = end_visited[node] + [nbr,]
                        add_end(nbr)
        else:
            cur_node = start_edge.copy()
            start_edge.clear()
            for node in cur_node:
                for nbr in adj[node]:
                    if nbr in end_visited:
                        ### Found path
                        # Combined routes and return
                        path = start_visited[node] + end_visited[nbr][::-1]
                        return path
                    elif nbr not in start_visited:
                        # Extend start ring
                        start_visited[nbr] = start_visited[node] + [nbr,]
                        add_start(nbr)
    return None

def all_shortest_path_weighted_single(G, start):
    raise NotImplementedError
    # Dijkstra's algorithim
    
def all_shortest_path_weighted(G):
    raise NotImplementedError

def shortest_path_weighted(G, start, end):
    raise NotImplementedError

def path_connection(G, start, end):
    raise NotImplementedError


#### Graph Metrics ####
## Clustering Coefficient ## 

def num_triangles(G, node=None):
    '''
    Finds the number of triangles around a given node or all nodes in the graph
    '''
    adj = G.adj
    if node is not None:
        num_triag = 0
        nbrhood = set(adj[node])
        # Make for into generator tpye layout
        for nbr in nbrhood:
            num_triag += len(nbrhood & set(adj[nbr]))

        return num_triag // 2
    else:
        # Will be double checking some triangles, might be able to skip ever other node 
        # while making sure not to repeat
        num_triag = {}
        for node in G:
            triangles = 0
            nbrhood = set(adj[node])
            # Make for into generator tpye layout
            for nbr in nbrhood:
                triangles += len(nbrhood & set(adj[nbr]))
            num_triag[node] = triangles //2
        return num_triag

def local_cluster_coef(G, node=None):
    if node is not None:
        temp = len(G.adj[node])
        return 2*num_triangles(G, node)/(temp*(temp-1))
    else:
        cluster_coefs = {}
        for node in G:
            temp = len(G.adj[node])
            cluster_coefs[node] = 2 * num_triangles(G, node) / (temp*(temp-1))
        return cluster_coefs

def global_cluster_coef(G):
    total = 0
    for node in G:
            temp = len(G.adj[node])
            total += 2*num_triangles(G, node)/(temp*(temp-1))
    return total/len(G)

def num_squares(G, node=None):
    adj = G.adj
    if node is not None:
        count = 0
        nbrhood = adj[node]
        cousins = set()
        for nbr in nbrhood:
            temp = set(adj[nbr])
            count += len(temp)
            cousins = cousins | temp
        return count - len(cousins) - len(nbrhood) + 1

def square_clustering_coef(G, node=None):
    from itertools import combinations
    adj = G.adj
    if node is not None:
        nbrhood = adj[node]
        clustering = 0 
        tot_squares = 0
        pos_squares = 0
        for v, u in combinations(nbrhood, 2):
            num_squares = len(set(adj[v]) & set(adj[u])) - 1
            tot_squares += num_squares
            deg_sub = 1 + num_squares if u not in adj[v] else 2 + num_squares 
            pos_squares += (len(adj[v]) - deg_sub) * (len(adj[u]) - deg_sub) + num_squares
        if pos_squares > 0:
            clustering = tot_squares / pos_squares
            
        return clustering
        

def generalized_degree(G, node=None):
    raise NotImplementedError

def upperboudn_generalized_degree(G, node=None):
    raise NotImplementedError

# Cliques #

def find_cliques(G, max_size = 10):
    raise NotImplementedError


## Centralities ##

def degree_centrality(G, norm=True):
    adj = G.adj
    if norm:
        max_deg = len(G) - 1
        degrees = {node:len(adj[node])/max_deg for node in G}
    else:
        degrees = {node:len(adj[node]) for node in G}
    return degrees

def closenesss_centrality(G, node=None):
    raise NotImplementedError

## Laplacian ## 

def diagonal_matrix(G):
    import numpy as np
    adj = G.adj
    D = np.zeros((len(adj), len(adj)))

    for n, v in enumerate(adj):
        for m, u in enumerate(adj):
            D[n][n] = len(adj[v])

    return D

def adjacency_matrix(G):
    import numpy as np
    adj = G.adj
    A = np.zeros((len(adj), len(adj)))
    for n, v in enumerate(adj):
        for m, u in enumerate(adj):
            if m<n and v in adj[u]:
                A[n][m] = -1 
            elif m>n and u in adj[v]:
                A[n][m] = -1
    return A

def laplacian_matrix(G):
    import numpy as np
    adj = G.adj 
    L = np.zeros((len(adj), len(adj)))

    for n, v in enumerate(adj):
        for m, u in enumerate(adj):
            if v == u:
                L[n][m] = len(adj[v])
            elif m<n and v in adj[u]:
                L[n][m] = -1 
            elif m>n and u in adj[v]:
                L[n][m] = -1
    return L

## Drawing ##

def heatmap(G, heat_map):
    '''
    Draws a given graph with each node coloured based on some value between 0 and 1 provided as a dict.
    '''
    raise NotImplementedError