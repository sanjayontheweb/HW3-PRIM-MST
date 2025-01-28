import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        # Initialize empty array matrix
        self.mst = np.zeros_like(self.adj_mat)
        
        #Choose i = 0 as initial node
        visited = {0}
        neighbors = [(v,i,0) for i,v in enumerate(self.adj_mat[0]) if v != 0]
        heapq.heapify(neighbors)

        #Iterate n-1 times to add n-1 edges to self.mst
        n = self.adj_mat.shape[0]
        while len(visited) < n:
            edge_weight, out_node, in_node = heapq.heappop(neighbors)
            if out_node in visited:
                continue
            
            self.mst[in_node, out_node] = edge_weight
            self.mst[out_node, in_node] = edge_weight

            visited.add(out_node)
            [heapq.heappush(neighbors, (v,i,out_node)) for i,v in enumerate(self.adj_mat[out_node]) if (v != 0) and (i not in visited)]
