"""
Toy implementation of the Topical Affinity Propagation algorithm for estimating
topic-based social influence graphs, from Tang et al. 2009. "Social influence
analysis in large-scale networks." Conference on Knowledge Discovery & Data 
Mining 2009, June 28 - July 1, 2009, Paris, France. <http://bit.ly/1nOUh9e>.

TODO
----
* Convert print statements to logger debug.
* Write author_theta()

"""

import networkx as nx
import numpy as np
import random


def swap(u, v):
    """
    exchange the values of u and v
    """

    return v, u


def author_theta(papers, model):
    """
    Generates distributions over topics for authors, based on distributions over
    topics for their papers.
    """

    return theta


class TAPModel(object):
    def __init__(self, G, theta, damper=0.5):
        """
        
        Parameters
        ----------
        G : :class:`.nx.Graph()`
            Should have 'weight' attribute in [0.,1.].
        theta : array-like
            Should have shape (N, T), where N == len(G.nodes()) and T is the
            number of topics.
        """
        self.G = G  # TODO: should take G as an input.
        self.theta = theta

        # These dictionaries are indexed by node id and not necessarily 0-based.
        self.a = {}
        self.b = {}
        self.r = {}
        self.g = {}

        self.damper = damper  # This was not very clear in the paper.

        self.N = len(self.G.nodes())
        self.M = len(self.G.edges())

        print("Loaded graph with {0} nodes and {1} edges.".format(self.N, self.M))

        self.T = self.theta.shape[1]
        self.N_d = self.theta.shape[0]

        self.yold = {i: {k: 0 for k in range(self.T)} for i in self.G.nodes()}

        print(
            "Loaded distributions over {0} topics for {1} nodes.".format(
                self.T, self.N_d
            )
        )

        # 	1.1 calculate g(vi,yi,z)
        self._calculate_g()
        print("Calculated g")

        #   1.2 Eq8, calculate bz,ij
        self._calculate_b()
        print("Calculated b")

    def _calculate_g(self):
        """eq. 1"""
        for i in self.G.nodes():
            n = self.G.neighbors(i)
            self.g[i] = np.zeros((len(list(n)) + 1, self.T))

            sumin = np.zeros((self.T))
            sumout = np.zeros((self.T))

            for t, attr in self.G[i].items():
                this = int(t) - 1
                for k in range(self.T):
                    w = float(attr["weight"])
                    sumout[k] = sumout[k] + w * self.theta[this, k]

            for t, attr in self.G[i].items():
                for k in range(self.T):
                    w = float(attr["weight"])
                    this = int(i) - 1
                    sumin[k] = sumin[k] + w * self.theta[this, k]

                    # calculate y z, i=i ;; [n,] should be the last row.
                    self.g[i][len(list(n)), k] = sumin[k] / (sumin[k] + sumout[k])

            j = 0
            for t, attr in self.G[i].items():
                for k in range(self.T):
                    w = float(attr["weight"])
                    this = int(t) - 1
                    self.g[i][j, k] = w * self.theta[this, k] / (sumin[k] + sumout[k])
                j += 1

    def _calculate_b(self):
        """eq. 8"""
        for i in self.G.nodes():
            n = self.G.neighbors(i)
            self.b[i] = np.zeros((len(list(n)) + 1, self.T))
            self.r[i] = np.zeros((len(list(n)) + 1, self.T))
            self.a[i] = np.zeros((len(list(n)) + 1, self.T))

            sum_ = np.zeros((self.T))

            for j in range(len(list(n)) + 1):  # +1 to include self.
                for k in range(self.T):
                    sum_[k] += self.g[i][j, k]
            for j in range(len(list(n)) + 1):
                for k in range(self.T):
                    self.b[i][j, k] = np.log(self.g[i][j, k] / sum_[k])

    def _update_r(self):
        """eq. 5"""

        for i in self.G.nodes():
            n = self.G.neighbors(i)

            firstmax = np.zeros((self.T))
            secondmax = np.zeros((self.T))
            temp = 0.0
            maxk = {}

            if len(list(n)) < 1:  # node has no neighbors
                for k in range(self.T):
                    self.r[i][0, k] = self.b[i][0, k]
            else:
                for k in range(self.T):
                    firstmax[k] = self.b[i][0, k] + self.a[i][0, k]

                    try:
                        secondmax[k] = self.b[i][1, k] + self.a[i][1, k]
                    except IndexError:
                        secondmax[k] = self.b[i][1, k]

                    maxk[k] = 0
                    if secondmax[k] > firstmax[k]:
                        firstmax[k], secondmax[k] = swap(firstmax[k], secondmax[k])
                        maxk[k] = 1

                for j in range(2, len(list(n)) + 1):
                    for k in range(self.T):
                        temp = self.a[i][j, k] + self.b[i][j, k]
                        if temp > secondmax[k]:
                            temp, secondmax[k] = swap(temp, secondmax[k])

                        if secondmax[k] > firstmax[k]:
                            firstmax[k], secondmax[k] = swap(firstmax[k], secondmax[k])
                            maxk[k] = j

                for j in range(len(list(n)) + 1):
                    for k in range(self.T):
                        if j == maxk[k]:
                            self.r[i][j, k] = (
                                (self.b[i][j, k] - secondmax[k]) * (1.0 - self.damper)
                            ) + (self.r[i][j, k] * self.damper)
                        else:
                            self.r[i][j, k] = (
                                (self.b[i][j, k] - firstmax[k]) * (1.0 - self.damper)
                            ) + (self.r[i][j, k] * self.damper)

    def _update_a(self):
        firstmax = {}
        secondmax = {}
        maxk = {}

        for j in self.G.nodes():
            firstmax[j] = np.zeros((self.T))
            secondmax[j] = np.zeros((self.T))

            maxk[j] = np.array([-1] * self.T)

            n = list(self.G.neighbors(j))

            # maxk[N] records the maximum value of min{r z, kj, 0}
            if len(n) < 1:
                for k in range(self.T):
                    firstmax[j][k] = 0.0

            else:
                neighbour = n[0]
                neighbor_list = list(self.G.neighbors(neighbour))
                pos = neighbor_list.index(j)

                for k in range(self.T):
                    try:
                        firstmax[j][k] = min(self.r[neighbour][pos, k], 0.0)
                    except IndexError:
                        firstmax[j][k] = 0

                    maxk[j][k] = neighbour

                if len(n) >= 2:
                    neighbour = n[1]
                    neighbor_list = list(self.G.neighbors(neighbour))
                    pos = neighbor_list.index(j)

                    for k in range(self.T):
                        try:
                            secondmax[j][k] = min(self.r[neighbour][pos, k], 0.0)
                        except IndexError:
                            secondmax[j][k] = 0

                        if secondmax[j][k] > firstmax[j][k]:
                            firstmax[j][k], secondmax[j][k] = swap(
                                firstmax[j][k], secondmax[j][k]
                            )
                            maxk[j][k] = neighbour

                    for i in range(2, len(n)):
                        neighbour = n[i]
                        neighbor_list = list(self.G.neighbors(neighbour))
                        pos = neighbor_list.index(j)

                        for k in range(self.T):
                            try:
                                temp = min(self.r[neighbour][pos, k], 0.0)
                            except IndexError:
                                temp = 0

                            if temp > secondmax[j][k]:
                                temp, secondmax[j][k] = swap(temp, secondmax[j][k])
                            if secondmax[j][k] > firstmax[j][k]:
                                firstmax[j][k], secondmax[j][k] = swap(
                                    firstmax[j][k], secondmax[j][k]
                                )
                                maxk[j][k] = neighbour

        for i in self.G.nodes():
            n = self.G.neighbors(i)
            for k in range(self.T):  # a_ii
                try:
                    self.a[i][len(list(n)), k] = firstmax[i][k]
                except IndexError:
                    self.a[i][k] = firstmax[i][k]

            for j in n:  # a_ij
                j_index = n.index(j)
                for k in range(self.T):
                    if i == maxk[j][k]:
                        use = secondmax[i][k]
                    else:
                        use = firstmax[i][k]
                        n_j = self.G.neighbors(j)
                    qwert = max(self.r[j][len(n_j), k], 0)
                    asdf = (-1 * min(self.r[j][len(n_j), k], 0)) - use
                    self.a[i][j_index, k] = (min(qwert, asdf) * (1.0 - self.damper)) + (
                        self.a[i][j_index, k] * self.damper
                    )

    def _check_convergence(self, nc):
        """
        Returns false if the ranking of influencing nodes hasn't changed in a while.
        """

        dc = 0
        for i in self.G.nodes():
            n = list(self.G.neighbors(i))
            for k in range(self.T):
                try:
                    firstmax = self.r[i][len(n), k] + self.a[i][len(n), k]
                except IndexError:
                    firstmax = self.r[i][0, k] + self.a[i][0, k]

                rep = -1

                for j in range(len(n)):
                    try:
                        temp = self.r[i][j, k] + self.a[i][j, k]
                    except IndexError:
                        temp = self.r[i][0, k] + self.a[i][0, k]

                    if temp > firstmax:
                        temp, firstmax = swap(temp, firstmax)
                        rep = j
                if rep == -1:
                    rep = i
                else:
                    rep = n[rep]

                if self.iteration >= 21:
                    if self.yold[i][k] != rep:
                        dc += 1

                self.yold[i][k] = rep

        if dc == 0:  # No change?
            nc += 1
        else:
            nc = 0

        cont = True
        if nc == 100:
            cont = False

        return nc, cont

    def _calculate_mu(self):
        self.MU = {}

        # Export
        for k in range(self.T):
            subg = nx.DiGraph()

            # Influence
            for i in self.G.nodes():
                n = list(self.G.neighbors(i))
                for j in self.G.nodes():
                    if j in n:
                        j_ = n.index(j)
                        j_neighbors = list(self.G.neighbors(j))
                        i_ = j_neighbors.index(i)

                        # Equation 9.
                        try:
                            j_i = 1.0 / (
                                1.0
                                + np.exp(-1.0 * (self.r[i][j_, k] + self.a[i][j_, k]))
                            )
                        except:
                            j_i = 1.0 / (
                                1.0 + np.exp(-1.0 * (self.r[i][0, k] + self.a[i][0, k]))
                            )

                        try:
                            i_j = 1.0 / (
                                1.0
                                + np.exp(-1.0 * (self.r[j][i_, k] + self.a[j][i_, k]))
                            )
                        except:
                            i_j = 1.0 / (
                                1.0 + np.exp(-1.0 * (self.r[j][0, k] + self.a[j][0, k]))
                            )

                        if j_i > i_j:  # Add only strongest edge.
                            subg.add_edge(j, i, weight=float(j_i))
                        else:
                            subg.add_edge(i, j, weight=float(i_j))

            # Theta
            constant = list(self.G.nodes())
            c = int(constant[0])
            for i in self.G.nodes():
                self.G.nodes[i]["theta"] = self.theta[int(i) - c, int(k)]

            self.MU[k] = subg

    def prime(self, alt_r, alt_a, alt_G):
        """
        Prime r and a with values from a previous model.
        
        Parameters
        ----------
        alt_r : dict
            { i: array-like [ j, k ] for i in G.nodes() }
            Must be from a model with the same topics.
        alt_a :dict
            { i: array-like [ j, k ] for i in G.nodes() }
            Must be from a model with the same topics.
        alt_G : :class:`.nx.Graph`
            Need not be the same shape as G, but node names must be consistent.
        """

        for i in alt_G.nodes():
            alt_n = list(alt_G.neighbors(i))
            if i in self.G.nodes():
                # alt_r and alt_a must be from a model with the same topics.
                assert alt_r[i].shape[1] == self.r[i].shape[1]
                assert alt_a[i].shape[1] == self.a[i].shape[1]

                n = list(self.G.neighbors(i))
                for j in alt_n:
                    if j in n:
                        j_ = n.index(j)
                        alt_j_ = alt_n.index(j)

                        for k in range(self.T):
                            try:
                                self.r[i][j_, k] = alt_r[i][alt_j_, k]
                            except IndexError:
                                self.r[i][0, k] = alt_r[i][0, k]

                            try:
                                self.a[i][j_, k] = alt_a[i][alt_j_, k]
                            except IndexError:
                                self.a[i][0, k] = alt_a[i][0, k]

    def write(self, target):
        for k in list(self.MU.keys()):
            nx.write_graphml(self.MU[k], "{0}_topic_{1}.graphml".format(target, k))

    def graph(self, k):
        return self.MU[k]

    def build(self):
        nc = 0
        self.iteration = 0.0
        cont = True

        while cont:
            self.iteration += 1
            self._update_r()
            self._update_a()
            nc, cont = self._check_convergence(nc)

        self._calculate_mu()

        self.write("./output/")


edgepath = "./sample/graph-16.edge"
distpath = "./sample/distribution.txt"

G = nx.Graph()

# Read in Graph data.
with open(edgepath, "r") as f:
    for line in f:
        edge = line.strip().split()
        G.add_edge(edge[0], edge[1], weight=edge[2])

# Topic distributions for nodes.
# TODO: should take this as an input.
with open(distpath, "r") as f:
    theta = np.loadtxt(f)

theta = np.random.rand(234, 10)

print("first model")
model = TAPModel(G, theta)
model.build()

alt_r, alt_a, alt_G = model.r, model.a, model.G

print("second model")
model2 = TAPModel(G, theta)
model2.prime(alt_r, alt_a, alt_G)
model2.build()
