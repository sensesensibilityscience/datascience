import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as wd
import pandas as pd
import plotly.express as px
from IPython.display import display
from inspect import signature
from graphviz import Digraph
from scipy.stats import pearsonr

class CausalNode:
  def __init__(self, vartype, func, name, causes=None, min=0, max=100):
    '''
    name: string, must be unique
    vartype: 'categorical', 'discrete', 'continuous'
    causes: (node1, ..., nodeN)
    func: f
    f is a function of N variables, matching the number of nodes and their types, returns a single number matching the type of this node
    self.network: {node_name: node, ...}, all nodes that the current node depends on
    '''
    # n_func_args = len(signature(func).parameters)
    # n_causes = 0 if causes == None else len(causes)
    # if n_func_args != n_causes:
    #   raise ValueError('The number of arguments in func does not match the number of causes.')
    self.name = name
    self.causes = causes
    self.func = func
    self.network = self.nodeDict()
    self.vartype = vartype
    self.min = min
    self.max = max

  def traceNetwork(self):
    '''
    Generates set of all nodes that current node depends on
    '''
    nodes = {self}
    if self.causes != None:
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
    '''
    data = {}
    while len(data) != len(self.network):
      for m, n in self.network.items():
        if m not in data.keys():
          if n.causes == None:
            if m not in fix.keys():
              data[m] = n.func()
            else:
              data[m] = fix[m]
          else:
            ready = True
            for c in n.causes:
              if c.name not in data.keys():
                ready = False
                break
            if ready:
              parents_val = [data[c.name] for c in n.causes]
              if m not in fix.keys():
                data[m] = n.func(*parents_val)
              else:
                data[m] = fix[m]
    return data

  def generate(self, n, intervene={}):
    '''
    Generates n data points. Returns dict of name, np.array(values) pairs
    intervene: {node_name: [type, other_args]}
    intervene format:
    ['fixed', val]
    ['range', start, end]
    ['range_rand', start, end]
    ['array', [...]] array size must be n
    '''
    fix_all = {} # {name: [val, ...], ...}
    for name, args in intervene.items():
      if args[0] == 'fixed':
        fix_all[name] = np.array([args[1] for i in range(n)])
      elif args[0] == 'range':
        fix_all[name] = np.linspace(args[1], args[2], n)
        if self.vartype == 'discrete':
          fix_all[name] = np.rint(fix_all[name])
      elif args[0] == 'range_rand':
        fix_all[name] = np.linspace(args[1], args[2], n)
        np.random.shuffle(fix_all[name])
        if self.vartype == 'discrete':
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

  def drawNetwork(self):
    g = Digraph(name=self.name)
    def draw_edges(node, g):
      if node.causes:
        for cause in node.causes:
          g.edge(cause.name, node.name)
          draw_edges(cause, g)
    draw_edges(self, g)
    return g

# Some functions for causal relations
def gaussian(mean, std):
  def f():
    return np.random.normal(mean, std)
  return f

def constant(x):
  def f():
    return x
  return f

def uniform(a, b):
  def f():
    return np.random.random()*(b-a) + a
  return f

def poisson(rate):
  def f():
    return np.random.poisson(lam=rate)
  return f

def choice(opts, weights=None, replace=True):
  def f():
    if weights == None:
      chosen = np.random.choice(opts, replace=replace)
    else:
      weights = np.array(weights)
      p = weights/sum(weights)
      chosen = np.random.choice(opts, p=p, replace=replace)
    return chosen
  return f

# Solves for the coefficients given a set of points
def solveLinear(*points):
  n = len(points)
  A = np.zeros((n, n))
  b = np.zeros(n)
  for i in range(n):
    A[i] = np.append(points[i][0:-1], 1)
    b[i] = points[i][-1]
  sol = np.linalg.solve(A, b)
  return sol[0:-1], sol[-1]

def linear(x1, y1, x2, y2, fuzz=0):
  M, c = solveLinear((x1, y1), (x2, y2))
  def f(x):
    return M[0]*x + c + np.random.normal(0, fuzz)
  return f

def linearFunc(x1, m1, c1, x2, m2, c2, func, fuzz=0, integer=False):
  M_m, c_m = solveLinear((x1, m1), (x2, m2))
  M_c, c_c = solveLinear((x1, c1), (x2, c2))
  def f(*args):
    x = args[-1]
    m = M_m[0]*x + c_m
    c = M_c[0]*x + c_c
    number = m*func(*args[0:-1]) + c + np.random.normal(0, fuzz)
    if integer:
      number = max(int(number), 0)
    return number
  return f

def dependentPoisson(*points):
  M, c = solveLinear(*points)
  def f(*args):
    rate = max(M@np.array(args) + c, 0)
    return np.random.poisson(lam=rate)
  return f

def dependentGaussian(x1, mean1, std1, x2, mean2, std2):
  M_mean, c_mean = solveLinear((x1, mean1), (x2, mean2))
  M_std, c_std = solveLinear((x1, std1), (x2, std2))
  def f(x):
    mean = M_mean[0]*x + c_mean
    std = max(M_std[0]*x + c_std, 0)
    return abs(np.random.normal(mean, std))
  return f

def categoricalLin(data): # data: {'category': (m, c), etc}
  def f(x, y): # y is the category, x is the input value
    return data[y][0] * x + data[y][1]
  return f

class InterveneOptions:
  def __init__(self, node, disabled=None):
    if disabled == None:
      self.disabled = [False, False, False]
    else:
      self.disabled = disabled
    self.name = node.name
    self.text = wd.Label(value=self.name, layout=wd.Layout(width='150px'))
    self.none = wd.RadioButtons(options=['None'], disabled=self.disabled[0], layout=wd.Layout(width='70px'))
    self.fixed = wd.RadioButtons(options=['Fixed'], disabled=self.disabled[1], layout=wd.Layout(width='70px'))
    self.fixed.index = None
    self.fixed_arg = wd.BoundedFloatText(disabled=True, layout=wd.Layout(width='70px'))
    self.range_visibility = 'hidden' if node.vartype == 'categorical' else 'visible'
    self.range = wd.RadioButtons(options=['Range'], disabled=self.disabled[2], layout=wd.Layout(width='70px', visibility=self.range_visibility))
    self.range.index = None
    self.range_arg1 = wd.BoundedFloatText(min=node.min, max=node.max, disabled=True, layout=wd.Layout(width='70px', visibility=self.range_visibility))
    self.range_arg2 = wd.BoundedFloatText(min=node.min, max=node.max, disabled=True, layout=wd.Layout(width='70px', visibility=self.range_visibility))
    self.range_rand = wd.Checkbox(description='Randomise Order', disabled=True, indent=False, layout=wd.Layout(visibility=self.range_visibility))
    self.none.observe(self.none_observer, names=['value'])
    self.fixed.observe(self.fixed_observer, names=['value'])
    self.range.observe(self.range_observer, names=['value'])
    self.box = wd.HBox([self.text, self.none, self.fixed, self.fixed_arg, self.range, self.range_arg1, self.range_arg2, self.range_rand])

  def display(self):
    display(self.box)

  def greyAll(self):
    self.none.disabled = True
    self.fixed.disabled = True
    self.fixed_arg.disabled = True
    self.range.disabled = True
    self.range_arg1.disabled = True
    self.range_arg2.disabled = True
    self.range_rand.disabled = True

  def applyIntervene(self, intervene):
    if intervene[0] == 'fixed':
      self.fixed.index = 0
      self.fixed_arg.value = intervene[1]
    elif intervene[0] == 'range':
      self.range.index = 0
      self.range_arg1.value = intervene[1]
      self.range_arg2.value = intervene[2]
    elif intervene[0] == 'range_rand':
      self.range.index = 0
      self.range_arg1.value = intervene[1]
      self.range_arg2.value = intervene[2]
      self.range_rand.value = True

  # Radio button .index = None if off, .index = 0 if on
  def none_observer(self, sender):
    if self.none.index == 0:
      self.fixed.index = None
      self.fixed_arg.disabled = True
      self.range.index = None
      self.range_arg1.disabled = True
      self.range_arg2.disabled = True
      self.range_rand.disabled = True

  def fixed_observer(self, sender):
    if self.fixed.index == 0:
      self.none.index = None
      self.fixed_arg.disabled = False
      self.range.index = None
      self.range_arg1.disabled = True
      self.range_arg2.disabled = True
      self.range_rand.disabled = True

  def range_observer(self, sender):
    if self.range.index == 0:
      self.none.index = None
      self.fixed.index = None
      self.fixed_arg.disabled = True
      self.range_arg1.disabled = False
      self.range_arg2.disabled = False
      self.range_rand.disabled = False

class GroupSettings:
  def __init__(self, node, disabled, show='all'):
    self.node = node
    self.group_name_text = wd.Label(value='Name the Group', layout=wd.Layout(width='150px'))
    self.group_name = wd.Text(layout=wd.Layout(width='150px'))
    self.group_name_box = wd.HBox([self.group_name_text, self.group_name])
    self.N_input_text = wd.Label(value='Number of Samples', layout=wd.Layout(width='150px'))
    self.N_input = wd.BoundedIntText(value=100, min=1, max=1000, layout=wd.Layout(width='70px'))
    self.N_input_box = wd.HBox([self.N_input_text, self.N_input])
    self.opts_single = {}
    for m, n in node.network.items():
      if show != 'all' and m not in show:
        continue
      d = None
      if m in disabled:
        d = [False, True, True]
      self.opts_single[m] = InterveneOptions(n, disabled=d)
    to_display = [self.group_name_box, self.N_input_box]
    for m in sorted(self.opts_single.keys()): # Ensure alphabetical display order
      to_display.append(self.opts_single[m].box)
    self.box = wd.VBox(to_display, layout=wd.Layout(margin='0 0 20px 0'))

  def display(self):
    display(self.box)

  def append(self, *args):
    to_display = self.box.children + args
    self.box.children = to_display

  def remove(self):
    to_display = self.box.children[0:-2]
    self.box.children = to_display

  def greyAll(self):
    self.group_name.disabled = True
    self.N_input.disabled = True
    for m, o in self.opts_single.items():
      o.greyAll()

  def applyIntervene(self, config):
    '''
    config: dictionary imported from json file
    {
    'name': ,
    'N': ,
    'intervene': {
      'node_name': ['type', args...],
      ...
    }
    }
    '''
    self.group_name.value = config['name']
    self.N_input.value = config['N']
    for m, i in config['intervene'].items():
      if m == self.node.name:
        continue
      self.opts_single[m].applyIntervene(i)

class PlotSettings:
  def __init__(self, names):
    self.names = names
    self.varsx = wd.RadioButtons(options=names, layout=wd.Layout(width='200px'))
    self.varsy = wd.RadioButtons(options=names, layout=wd.Layout(width='200px'))
    self.colx = wd.VBox([wd.Label(value='x-Axis Variable'), self.varsx])
    self.coly = wd.VBox([wd.Label(value='y-Axis Variable'), self.varsy])
    self.hbox = wd.HBox([self.colx, self.coly])
    self.box = wd.VBox([self.hbox], layout=wd.Layout(margin='0 0 20px 0'))

  def display(self):
    display(self.box)

  def chosen(self):
    namex = self.names[self.varsx.index]
    namey = self.names[self.varsy.index]
    return (namex, namey)

  def append(self, *args):
    to_display = self.box.children + args
    self.box.children = to_display

  def remove(self):
    to_display = self.box.children[0:-2]
    self.box.children = to_display

class Experiment:
  def __init__(self, node):
    self.node = node
    self.data = {} # {group_name: {node_name: [val, ...], ...}, ...}
    self.group_names = []

  def setting(self, disabled=[], show='all'):
    '''
    disabled: array of names
    '''
    settings = [GroupSettings(self.node, disabled, show=show)]
    add_group = wd.Button(description='Add Another Group')
    submit = wd.Button(description='Perform Experiment')
    settings[0].display()
    settings[0].append(add_group, submit)
    add_group.on_click(self.addGroup(settings, disabled))
    submit.on_click(self.doExperiment(settings))

  def fixedSetting(self, config, show='all'):
    '''
    For demonstrating a preset experiment, disable all options and display the settings
    config: array of intervenes
    '''
    settings = []
    for c in config:
      s = GroupSettings(self.node, disabled=[], show=show)
      s.applyIntervene(c)
      s.greyAll()
      s.display()
      settings.append(s)
    self.doExperiment(settings)()

  def addGroup(self, settings, disabled, show='all'):
    def f(sender):
      buttons = settings[-1].box.children[-2:]
      settings[-1].remove() # Remove the buttons from previous group
      settings.append(GroupSettings(self.node, disabled=disabled, show=show))
      settings[-1].append(*buttons) # Add buttons to the newly added group
      settings[-1].display()
    return f

  def generateIntervene(self, opts):
    intervene = {} # Gets passed to self.generate
    for name, opt in opts.items():
      if opt.none.index == 0: # None is deselected, 0 is selected
        continue
      elif opt.fixed.index == 0:
        intervene[name] = ['fixed', opt.fixed_arg.value]
      elif opt.range.index == 0:
        if opt.range_rand.value:
          intervene[name] = ['range_rand', opt.range_arg1.value, opt.range_arg2.value]
        else:
          intervene[name] = ['range', opt.range_arg1.value, opt.range_arg2.value]
    return intervene

  def doExperiment(self, settings):
    def f(sender=None):
      names = []
      for s in settings:
        name = s.group_name.value
        names.append(name)
        n = s.N_input.value
        intervene = self.generateIntervene(s.opts_single)
        self.data[name] = self.node.generate(n, intervene=intervene)
      self.group_names = names
    return f

  def plotSetting(self, show='all'):
    node_names = sorted([*self.node.network]) if show == 'all' else sorted(show)
    settings = [PlotSettings(node_names)]
    add_plot = wd.Button(description='Add Another Plot')
    submit = wd.Button(description='Draw Plots')
    settings[0].display()
    settings[0].append(add_plot, submit)
    add_plot.on_click(self.addPlot(settings, show=show))
    submit.on_click(self.plot(settings))

  def addPlot(self, settings, show):
    def f(sender):
      buttons = settings[-1].box.children[-2:]
      settings[-1].remove() # Remove the buttons from previous group
      node_names = sorted(self.node.network.keys())
      names = node_names if show == 'all' else show
      settings.append(PlotSettings(names))
      settings[-1].append(*buttons) # Add buttons to the newly added group
      settings[-1].display()
    return f

  def plot(self, settings):
    def f(sender):
      for s in settings:
        for name in self.group_names:
          x, y = s.chosen()[0], s.chosen()[1]
          self.choosePlot(x, y, name)
          if name:
            plt.title(name + ": " + x + ' vs. ' + y)
          else:
            plt.title(x + ' vs. ' + y)
          plt.xlabel(x)
          plt.ylabel(y)
          plt.show()
          r = pearsonr(self.data[name][x], self.data[name][y])
          print("Correlation (r): ", '{0:#.3f}'.format(r[0]))
          print("P-value: ", '{0:#.3g}'.format(r[1]))
    return f

  def choosePlot(self, x, y, name):
    """x and y are the names of the variables to plot on the x and y axes
    name is the name of the group in the experiment
    Returns the most appropriate plot type for those two variables"""
    xType, yType = self.node.nodeDict()[x].vartype, self.node.nodeDict()[y].vartype
    xData, yData = self.data[name][x], self.data[name][y]
    if xType == 'categorical' and yType != 'categorical':
      plot = plt.hist(yData)
    elif xType != 'categorical' and yType == 'categorical':
      plot = plt.hist(xData)
    elif xType == 'continuous' and yType == 'continuous':
      plot = plt.scatter(xData, yData, c='purple')
    else:
      heatmap = plt.hist2d(xData, yData, bins=30, cmap=plt.cm.BuPu)
      plt.colorbar(heatmap[3])

  def plotOrchard(self, name, gradient=None):
    """Takes in the name of the group in the experiment and the name of the 
    variable used to create the color gradient"""
    fig = px.scatter(self.data[name], x="x", y="y", color=gradient, title='Orchard Layout:' + name, hover_data=self.data[name].keys())
    fig.update_layout({'height':650, 'width':650})
    fig.show()

# Uniformly distributed from 0m to 1000m
x_node = CausalNode('continuous', uniform(0, 1000), name='x', min=0, max=1000)
y_node = CausalNode('continuous', uniform(0, 1000), name='y', min=0, max=1000)
# Gaussian+absolute value, more wind in south
wind_node = CausalNode('continuous', lambda x,y: dependentGaussian(0, 2, 5, 1000, 10, 10)(x) + dependentGaussian(0, 6, 3, 1000, 2, 4)(x), name='Wind Speed', causes=[x_node, y_node], min=0, max=40)
suppliment_node = CausalNode('categorical', constant('Water'), name='Suppliment')
fertilizer_node = CausalNode('continuous', gaussian(10, 2), 'Fertilizer', min=0, max=20)
suppliment_soil_effects = {'Water': (1, 0), 'Kombucha': (0.6, -5), 'Milk': (1.2, 10), 'Tea': (0.7, 0)}
# Fertilizer improves soil, kombucha destroys it
soil_node = CausalNode('continuous', lambda x, y: categoricalLin(suppliment_soil_effects)(linear(0, 10, 20, 100, fuzz=5)(x), y), 'Soil Quality', causes=[fertilizer_node, suppliment_node], min=0, max=100)
suppliment_bees_effects = {'Water': (1, 0), 'Kombucha': (1.5, 0), 'Milk': (1, 0), 'Tea': (1.3, 0)}
# Beehive in north, bees avoid wind, love kombucha
bees_node = CausalNode('discrete', lambda x, y, z: categoricalLin(suppliment_bees_effects)(dependentPoisson((0, 0, 250), (500, 30, 10), (0, 30, 40))(x, y), z), name='Number of Bees', causes=[x_node, wind_node, suppliment_node], min=0, max=300)
# Bees and good soil improve fruiting
fruit_node = CausalNode('discrete', dependentPoisson((0, 0, 0), (100, 200, 40), (100, 50, 16)), name='Number of Fruits', causes=[soil_node, bees_node])
# fruit_node.drawNetwork()