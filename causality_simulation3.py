import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as wd
import pandas as pd
from IPython.display import display, update_display, Javascript, HTML
from inspect import signature
from graphviz import Digraph
import scipy.stats as sp
import plotly.express as px
import plotly.graph_objects as go
import warnings
import re

# display(HTML('''<style>
#     [title="Assigned samples:"] { min-width: 150px; }
# </style>'''))

def dialog(title, body, button):
    display(Javascript("require(['base/js/dialog'], function(dialog) {dialog.modal({title: '%s', body: '%s', buttons: {'%s': {}}})});" % (title, body, button)))

class CausalNetwork:
    def __init__(self):
        self.nodes = {} # 'node_name': node
        self.init_data = {}

    def addNode(self, node, init=False):
        if node.name in self.nodes.keys(): # check for duplicate nodes
            raise ValueError('A node with the same name %s already exists!' % node['name'])
        self.nodes[node.name] = node
        if init:
            self.init_data[node.name] = [] # to be populated by init()

    def addCause(self, effect, cause):
        self.nodes[effect].setCause(func=identity, causes=[cause])

    # to be run at the beginning of each notebook
    def init(self, data):
        for n in self.nodes:
            self.replacePlaceholders(n)
        for name in self.init_data.keys(): # populate init_data with data, raises error if any key in init_data is not present in data
            self.init_data[name] = data[name]
        # TODO: validate causal network has no single-direction loops (other loops are allowed)
        # validate init_data matches with the nodes that have .causes == None

    # TODO: is this even necessary??
    def setRoot(self, node):
        self.root_node = node

    # replaces all PlaceholderNodes of a node's causes by the actual node searched by name
    def replacePlaceholders(self, node):
        if isinstance(node, PlaceholderNode):
            raise ValueError('A PlaceholderNode cannot be passed directly to replacePlaceholders. Use it on the parent instead.')
        else:
            for i, n in enumerate(node.causes):
                if isinstance(n, PlaceholderNode):
                    if n.name not in self.nodes:
                        raise ValueError("Node %s doesn't exist in the causal network!" % n.name)
                    node.causes[i] = self.nodes[n.name] # replace
                else:
                    self.replacePlaceholders(n) # recurse down the tree

class CausalNode:
    def __init__(self, name=None):
        self.name = name if name else id(self) # given name or use unique id
        self.causes = None # array of node type arguments to f. None means it's an init node
        self.func = None # only takes positional arguments

    def setCause(self, func, causes):
        self.causes = causes
        self.func = func

    def traceNetwork(self):
        '''
        Generates set of all nodes that current node depends on
        '''
        nodes = {self}
        for c in self.causes:
            nodes = nodes.union(c.traceNetwork())
        return nodes

    def nodeDict(self):
        '''
        Generates a dictionary of name, node pairs for easier lookup of nodes by name
        '''
        nodes = self.traceNetwork()
        network = {}
        for n in nodes:
            network[n.name] = n
        return network

    def generateSingle(self, fix={}):
        '''
        Generates a single multidimensional data point. Returns dict of name, value pairs
        fix: {node_name: val, ...}
        returns data: {node_name: [val, val, ...], ...}
        '''
        data = {}
        node_dict = self.nodeDict()
        while len(data) != len(node_dict):
            for m, n in node_dict.items():
                if m not in data.keys():
                    if n.causes == None: # must be an init node
                        data[m] = fix[m]
                    else: # apply .func onto the values from .causes
                        ready = True # checks if all its causes have been evaluated
                        for c in n.causes:
                            if c.name not in data.keys():
                                ready = False
                                break
                        if ready: # evaluate this node using values from its .causes
                            parents_val = [data[c.name] for c in n.causes]
                            if m not in fix.keys():
                                data[m] = n.func(*parents_val)
                            else: # this case is way down here because we want to evaluate all the otherwise relevant nodes even if they're not used due to an intervention
                                data[m] = fix[m]
        return data

    # TODO move/add generate to CausalNetwork for all nodes, in order to be root_node agnostic
    def generate(self, n, intervention={}):
        '''
        Generates n data points. Returns dict of name, np.array(values) pairs
        intervention: {node_name: [type, other_args]}
        intervention format:
        ['fixed', val] (val could be number or name of category)
        ['range', start, end]
        ['array', [...]] array size must be n
        '''
        fix_all = {} # {name: [val, ...], ...}
        for name, args in intervention.items():
            if args[0] == 'fixed':
                fix_all[name] = np.array([args[1] for i in range(n)])
            elif args[0] == 'range':
                fix_all[name] = np.random.permutation(np.linspace(args[1], args[2], n))
                if isinstance(self, DiscreteNode):
                    fix_all[name] = np.rint(fix_all[name])
            elif args[0] == 'array':
                fix_all[name] = np.array(args[1])
        fixes = [None] * n # Convert to [{name: val, ...}, ...]
        for i in range(n):
            fixes[i] = {}
            for name, arr in fix_all.items():
                fixes[i][name] = arr[i]
        data_dicts = [self.generateSingle(fix=fix) for fix in fixes]
        data = {}
        for name in self.network:
            data[name] = np.array([d[name] for d in data_dicts])
        return pd.DataFrame(data)

class Experiment:
    def __init__(self, network):
        self.node = network.root_node
        l = []
        for key, arr in init_data.items():
            l.append(len(arr))
        if max(l) != min(l):
            raise ValueError('Every array in init_data must have the same length.')
        self.init_data = init_data
        if set(init_data.keys()) != set(network.init_attr):
            raise ValueError("init_data doesn't match the causal network's init_attr.")
        self.N = l[0] # Sample size
        self.data = {} # {group_name: {node_name: [val, ...], ...}, ...}
        self.assigned = False
        self.done = False
        self.p = None

# placeholder node that only has a name, used to look up the actual node in the CausalNetwork
class PlaceholderNode(CausalNode):
    pass

class ContinuousNode(CausalNode):
    def __init__(self, name=None, min=0, max=100):
        super().__init__(name=name)
        self.min = min
        self.max = max

class DiscreteNode(CausalNode):
    def __init__(self, name=None, min=0, max=100):
        super().__init__(name=name)
        self.min = min
        self.max = max

class CategoricalNode(CausalNode):
    def __init__(self, name=None, categories=[]):
        super().__init__(name=name)
        self.categories = categories

# returns a placeholder CausalNode whose name can be used to look up the actual node in the CausalNetwork
def node(name):
    return PlaceholderNode(name=name)

# lifts a function that can takes any non-node inputs into an equivalent function in which the inputs can also be nodes. lift(func) cannot take arrays of nodes as input
def lift(func):
    def liftedFunc(*args, **kwargs): # should have exactly the same format as the unlifted func
        n_args = len(args) # number of positional arguments
        names = list(kwargs.keys())
        vargs = list(args) + [kwargs[name] for name in names] # combine all arguments into positional arguments
        nargs = [v for v in vargs if isinstance(v, CausalNode)] # only node type arguments
        node_inds = [i for i, v in enumerate(vargs) if isinstance(v, CausalNode)] # positions in vargs of all node type arguments. node_inds[i] = position of nargs[i] in vargs
        node_inds_lookup = {j: i for i, j in enumerate(node_inds)} # given index in vargs, gives the index in nargs
        def f(*fnargs): # wrapper of the unlifted func, only given non-node positional arguments that are intended to be lifted to node types. fnargs has the same structure as nargs
            # here reconstruct fargs and fkwargs from fnargs and the non-node type args/kwargs
            fargs = [(fnargs[node_inds_inv[i]] if isinstance(v, CausalNode) else args[i]) for i, v in enumerate(args)] # replaces the nodes in args by the corresponding values in fnargs
            fkwargs = {n: (fnargs[node_inds_lookup[i+n_args]] if isinstance(kwargs[n], CausalNode) else kwargs[n]) for i, n in enumerate(names)} # replaces the nodes in fwargs by the corresponding values in fnargs
            return func(*fargs, **fkwargs)
        y = CausalNode()
        y.setCause(f, nargs)
        return y
    return liftedFunc

@lift
def identity(x):
    return x

@lift
def toInt(x):
    return round(x)

@lift
def bound(x, floor=-np.inf, ceil=np.inf):
    return np.min(np.max(x, floor), ceil)

@lift
def sum(*args):
    total = 0
    for a in args:
        total += a
    return total

@lift
def normal(mean, stdev):
    return np.random.normal(mean, stdev)

@lift
def poisson(rate):
    return np.random.poisson(lam=rate)

@lift
def dist(*args, point):
    return np.linalg.norm(np.array(args)-np.array(point))

@lift
def linear(*args, m=1, c=0, points=None):
    if points:
        m, c = solveLinear(points)
    elif len(args) == 1:
        m = [m]
    return np.array(m)@np.array(args)+c

@lift
def categoricalLinear(x, category, params):
    m, c = params[category]
    return m*x+c

# consider m@x+c=y where m,x are arrays and c,y are numbers. returns m,c given an array of points (x1, x2, ..., y)
def solveLinear(points):
    n = len(points)
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        A[i] = np.append(points[i][0:-1], 1)
        b[i] = points[i][-1]
    sol = np.linalg.solve(A, b)
    return sol[0:-1], sol[-1]

'''
truffula new format
'''
truffula = CausalNetwork()
truffula.addNode(ContinuousNode(name='Latitude', min=0, max=1000), init=True)
truffula.addNode(ContinuousNode(name='Longitude', min=0, max=1000), init=True)
truffula.addNode(ContinuousNode(name='Wind speed', min=0, max=40))
truffula.addCause(effect='Wind speed',
    cause=bound(
        normal(
            mean=linear(
                dist(node('Latitude'), node('Longitude'), (3000, 2000)), # absolute distance from a special point
                points=[(2236, 20), (3605, 3)] # mean speed=20 if close to point, =3 if far from point
                ),
            stdev=linear(
                dist(node('Latitude'), node('Longitude'), (1010, 800)),
                points=[(2236, 10), (3605, 3)] # stdev=10 if close to point, =3 if far from point
                )
            ),
        floor=0)
    )
truffula.addNode(CategoricalNode(name='Supplement', categories=['Water', 'Kombucha', 'Milk', 'Tea']))
truffula.addNode(ContinuousNode(name='Fertilizer', min=0, max=20))
truffula.addNode(ContinuousNode(name='Soil quality', min=0, max=100))
truffula.addCause(effect='Soil quality',
    cause=categoricalLinear(
        sum(
            linear(
                node('Fertilizer'),
                points=[(0, 10), (20, 100)] # quality=10 if 0 fertilizer, =100 if 20 fertilizer
                ),
            normal(mean=0, stdev=5) # formerly "fuzz"
            ),
        category=node('Supplement'),
        params={'Water': (1, 0), 'Kombucha': (0.6, -5), 'Milk': (1.2, 10), 'Tea': (0.7, 0)} # (m, c), m=multiplier, c=additive constant
        )
    )
truffula.addNode(DiscreteNode(name='Number of bees', min=0, max=300))
truffula.addCause(effect='Number of bees',
    cause=toInt(
        categoricalLinear(
            poisson( # rate depends on distance to bee hive and wind speed
                linear(
                    dist(node('Latitude'), node('Longitude'), (-50, 30)), # distance from bee hive
                    node('Wind speed'),
                    points=[(30, 0, 250), (1000, 30, 10), (30, 30, 40)] # rate=250 if close to hive with no wind, =10 if far from hive with high wind, =40 if close to hive with high wind
                    )
                ),
            category=node('Supplement'),
            params={'Water': (1, 0), 'Kombucha': (1.3, 0), 'Milk': (1, 0), 'Beer': (0.2, 0)} # effect of supplements on bees
            )
        )
    )
truffula.addNode(DiscreteNode(name='Number of fruits', min=0, max=100))
truffula.addCause(effect='Number of fruits',
    cause=poisson(
        linear(
            node('Soil quality'), node('Number of bees'),
            points=[(0, 0, 0), (100, 200, 28), (100, 50, 16)] # rate=0 if poor soil and no bees, =28 if good soil and high bees, =16 if good soil and some bees
            )
        )
    )
truffula.setRoot(node='Number of fruits')