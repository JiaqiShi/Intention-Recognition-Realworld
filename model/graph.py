import numpy as np

def connect2adjacency(connects, num_joint):
    adjacency = np.zeros((num_joint, num_joint))
    for (i, j) in connects:
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    return adjacency

def normalize_graph(adjacency):
    Dl = np.sum(adjacency, 0)
    num_joint = adjacency.shape[0]
    Dn = np.zeros((num_joint, num_joint))
    for i in range(num_joint):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, adjacency), Dn)
    return DAD

class Graph():
    ''' The Graph to model the 3D skeletal data

    Args:
        strategy: Select one of following adjacency strategies
        - unilabel
        - distance
        - spatial
        - part
        max_dis_connect: max connection distance
    '''
    def __init__(self, strategy='spatial', max_dis_connect=1):
        self.strategy = strategy
        self.max_dis_connect = max_dis_connect

        self.get_edge()
        self.get_adjacency()

    def get_edge(self):
        self.center = 2
        self.num_joint = 17
        self_connect = [(i, i) for i in range(self.num_joint)]
        self.self_connect = self_connect

        if self.strategy == 'part':
            head = [(8, 9), (9, 10), (8, 11), (8, 14)]
            lefthand = [(14, 15), (15, 16)]
            righthand = [(11, 12), (12, 13)]
            torso = [(0, 7), (7, 8), (8, 11), (8, 14), (0, 1), (0, 4)]
            leftleg = [(1, 2), (2, 3)]
            rightleg = [(4, 5), (5, 6)]
            self.parts = [head, lefthand, righthand, torso, leftleg, rightleg]
        else:
            neighbor_connect = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                                (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                                (8, 14), (14, 15), (15, 16)]
            self.edge = self_connect + neighbor_connect

    def get_adjacency(self):
        if self.strategy == 'part':
            A = []
            A.append(connect2adjacency(self.self_connect, self.num_joint))
            for p in self.parts:
                A.append(normalize_graph(connect2adjacency(p, self.num_joint)))
            self.A = np.stack(A)
        else:
            adjacency = np.zeros((self.num_joint, self.num_joint))
            for i, j in self.edge:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
            dis_matrix = np.zeros((self.num_joint, self.num_joint)) + np.inf
            trans_matrix = [
                np.linalg.matrix_power(adjacency, p)
                for p in range(self.max_dis_connect + 1)
            ]
            N = np.zeros((self.num_joint, self.num_joint))
            for dis in range(self.max_dis_connect, -1, -1):
                dis_matrix[trans_matrix[dis] > 0] = dis
                N[trans_matrix[dis] > 0] = 1
            N = N / np.sum(N, 0)

            if self.strategy == 'unilabel':
                self.A = N[np.newaxis, :]
            elif self.strategy == 'distance':
                A = np.zeros(
                    (self.max_dis_connect + 1, self.num_joint, self.num_joint))
                for dis in range(self.max_dis_connect + 1):
                    A[dis][dis_matrix == dis] = N[dis_matrix == dis]
                self.A = A
            elif self.strategy == 'spatial':
                A = []
                for dis in range(self.max_dis_connect + 1):
                    root = np.zeros((self.num_joint, self.num_joint))
                    close = np.zeros((self.num_joint, self.num_joint))
                    further = np.zeros((self.num_joint, self.num_joint))
                    for i in range(self.num_joint):
                        for j in range(self.num_joint):
                            if dis_matrix[i, j] == dis:
                                if dis_matrix[i, self.center] == dis_matrix[
                                        j, self.center]:
                                    root[i, j] = N[i, j]
                                elif dis_matrix[i, self.center] < dis_matrix[
                                        j, self.center]:
                                    close[i, j] = N[i, j]
                                else:
                                    further[i, j] = N[i, j]
                    if dis == 0:
                        A.append(root)
                    else:
                        A.append(root + close)
                        A.append(further)
                self.A = np.stack(A)
            else:
                raise ValueError('[Error] Strategy not existing.')
