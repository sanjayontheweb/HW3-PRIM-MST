import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances

# Written with the aid of ChatGPT

def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # check correct # of edges
    assert np.count_nonzero(mst) == 2 * (adj_mat.shape[0] - 1)

    # Check for connectedness using BFS
    visited = set()
    def bfs_connected(start_node):
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                neighbors = np.where(mst[node] > 0)[0]
                queue.extend([n for n in neighbors if n not in visited])

    bfs_connected(0)  # Start BFS from node 0
    assert len(visited) == adj_mat.shape[0], 'Proposed MST is not connected'

    # Check for cycles using BFS
    def bfs_cycle(start_node):
        queue = [(start_node, -1)]  # (node, parent)
        visited = set()
        while queue:
            node, parent = queue.pop(0)
            if node not in visited:
                visited.add(node)
                for neighbor in np.where(mst[node] > 0)[0]:
                    if neighbor not in visited:
                        queue.append((neighbor, node))
                    elif neighbor != parent:
                        return True  # Cycle detected
        return False

    assert not bfs_cycle(0), 'Proposed MST contains a cycle'





def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    #Check empty matrix
    matrix = np.array([])
    with pytest.raises(ValueError):
        g = Graph(matrix)
        g.construct_mst()
        pass 
    

    #Check non-square matrix
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        g = Graph(matrix)
        g.construct_mst()
        pass 


    # Mock adjacency matrix
    adj_mat = np.array([
        [0, 2, 3, 0, 0],
        [2, 0, 15, 2, 0],
        [3, 15, 0, 0, 13],
        [0, 2, 0, 0, 9],
        [0, 0, 13, 9, 0]
    ])

    # Expected MST
    expected_mst = np.array([
        [0, 2, 3, 0, 0],
        [2, 0, 0, 2, 0],
        [3, 0, 0, 0, 0],
        [0, 2, 0, 0, 9],
        [0, 0, 0, 9, 0]
    ])

    # Expected weight of the MST
    expected_weight = 16  # 2 + 3 + 2 + 9 = 16

    # Construct MST using the Graph class
    g = Graph(adj_mat)
    g.construct_mst()

    # Assert that the constructed MST matches the expected MST
    assert np.allclose(g.mst, expected_mst), "MST does not match the expected result."

    # Check the total weight of the MST
    total_weight = np.sum(g.mst) / 2  # Sum of edges (undirected graph)
    assert abs(total_weight - expected_weight) < 1e-6, "MST weight is incorrect."

