import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as wd
import pandas as pd
from IPython.display import display, update_display
from inspect import signature
from graphviz import Digraph
import scipy.stats as sp
import plotly.express as px
import plotly.graph_objects as go

class CausalNode:
    def __init__(self, vartype, func, name, causes=None, min=0, max=100, categories=[]):
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
        self.categories = categories

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
        ['fixed', val] (val could be number or name of category)
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

class InterveneOptions:
    '''
    Line of radio button options for intervening in a single variable
    '''
    def __init__(self, node, disabled=None):
        if disabled == None:
            self.disabled = [False, False, False]
        else:
            self.disabled = disabled
        self.is_categorical = node.vartype == 'categorical'
        self.name = node.name
        self.text = wd.Label(value=self.name, layout=wd.Layout(width='180px'))
        self.none = wd.RadioButtons(options=['No intervention'], disabled=self.disabled[0], layout=wd.Layout(width='150px'))
        self.fixed = wd.RadioButtons(options=['Fixed'], disabled=self.disabled[1], layout=wd.Layout(width='70px'))
        self.fixed.index = None
        if self.is_categorical:
            fixed_arg = wd.Dropdown(options=node.categories, disabled=True, layout=wd.Layout(width='100px'))
        else:
            fixed_arg = wd.BoundedFloatText(disabled=True, layout=wd.Layout(width='70px'))
        self.fixed_arg = fixed_arg
        self.range_visibility = 'hidden' if self.is_categorical else 'visible'
        self.range = wd.RadioButtons(options=['Range'], disabled=self.disabled[2], layout=wd.Layout(width='70px', visibility=self.range_visibility))
        self.range.index = None
        self.range_arg1_text = wd.Label(value='from', layout=wd.Layout(visibility=self.range_visibility, width='30px'))
        self.range_arg1 = wd.BoundedFloatText(min=node.min, max=node.max, disabled=True, layout=wd.Layout(width='70px', visibility=self.range_visibility))
        self.range_arg2_text = wd.Label(value='to', layout=wd.Layout(visibility=self.range_visibility, width='15px'))
        self.range_arg2 = wd.BoundedFloatText(min=node.min, max=node.max, disabled=True, layout=wd.Layout(width='70px', visibility=self.range_visibility))
        self.range_rand = wd.Checkbox(description='Randomise Order', disabled=True, indent=False, layout=wd.Layout(visibility=self.range_visibility))
        self.none.observe(self.none_observer, names=['value'])
        self.fixed.observe(self.fixed_observer, names=['value'])
        self.range.observe(self.range_observer, names=['value'])
        self.box = wd.HBox([self.text, self.none, self.fixed, self.fixed_arg, self.range, self.range_arg1_text, self.range_arg1, self.range_arg2_text, self.range_arg2, self.range_rand])

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

    def grey(self, to_grey='all', grey_group_name=True, grey_N=True):
        self.group_name.disabled = grey_group_name
        self.N_input.disabled = grey_N
        if to_grey == 'all':
            to_grey = self.opts_single.items()
        for m, o in to_grey:
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
            # if m == self.node.name:
            #   continue
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
    def __init__(self, network):
        self.node = network.root_node
        self.data = {} # {group_name: {node_name: [val, ...], ...}, ...}
        self.group_names = []
        self.p = None

    def setting(self, disabled=[], show='all'):
        '''
        Let user design experiment
        disabled: array of names
        show: array of names
        '''
        settings = [GroupSettings(self.node, disabled, show=show)]
        add_group = wd.Button(description='Add Another Group')
        submit = wd.Button(description='Perform Experiment')
        settings[0].display()
        settings[0].append(add_group, submit)
        add_group.on_click(self.addGroup(settings, disabled, show=show))
        submit.on_click(self.doExperiment(settings))

    def fixedSetting(self, config, show='all'):
        '''
        For demonstrating a preset experiment, disable all options and display the settings
        config: array of intervenes
        show: array of names
        '''
        settings = []
        for c in config:
            s = GroupSettings(self.node, disabled=[], show=show)
            s.applyIntervene(c)
            s.grey()
            s.display()
            settings.append(s)
        self.doExperiment(settings)()

    def partialFixedSetting(self, config, show='all'):
        '''
        Let user design experiment, subject to constraints
        config: array of intervenes
        show: array of names
        '''
        settings = [GroupSettings(self.node, disabled, show=show)]

    def addGroup(self, settings, disabled, show='all'):
        def f(sender):
            buttons = settings[-1].box.children[-2:]
            settings[-1].remove() # Remove the buttons from previous group
            settings.append(GroupSettings(self.node, disabled=disabled, show=show))
            settings[-1].append(*buttons) # Add buttons to the newly added group
            settings[-1].display()
        return f

    def generateIntervene(self, opts):
        '''
        generates intervene from UI settings
        '''
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

    def doExperiment(self, settings, print=True):
        def f(sender=None):
            self.data = dict()
            names = []
            for s in settings:
                name = s.group_name.value
                names.append(name)
                n = s.N_input.value
                intervene = self.generateIntervene(s.opts_single)
                self.data[name] = self.node.generate(n, intervene=intervene)
            self.group_names = names
            if print:
                display(wd.Label(value='Data from experiment collected!'))
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

  # def plot(self, settings):
  #   def f(sender):
  #     for s in settings:
  #       for name in self.group_names:
  #         x, y = s.chosen()[0], s.chosen()[1]
  #         self.choosePlot(x, y, name)
  #         if name:
  #           plt.title(name + ": " + x + ' vs. ' + y)
  #         else:
  #           plt.title(x + ' vs. ' + y)
  #         plt.xlabel(x)
  #         plt.ylabel(y)
  #         plt.show()
  #         r = pearsonr(self.data[name][x], self.data[name][y])
  #         print("Correlation (r): ", '{0:#.3f}'.format(r[0]))
  #         print("P-value: ", '{0:#.3g}'.format(r[1]))
  #   return f

  # def choosePlot(self, x, y, name):
  #   """x and y are the names of the variables to plot on the x and y axes
  #   name is the name of the group in the experiment
  #   Returns the most appropriate plot type for those two variables"""
  #   xType, yType = self.node.nodeDict()[x].vartype, self.node.nodeDict()[y].vartype
  #   xData, yData = self.data[name][x], self.data[name][y]
  #   if xType == 'categorical' and yType != 'categorical':
  #     plot = plt.hist(yData)
  #   elif xType != 'categorical' and yType == 'categorical':
  #     plot = plt.hist(xData)
  #   elif xType == 'continuous' and yType == 'continuous':
  #     plot = plt.scatter(xData, yData, c='purple')
  #   else:
  #     heatmap = plt.hist2d(xData, yData, bins=30, cmap=plt.cm.BuPu)
  #     plt.colorbar(heatmap[3])

    def newPlot(self, show='all'):
        p = interactivePlot(self, show)
        self.p = p
        p.display()

    def plotOrchard(self, name, gradient=None, show='all'):
        """Takes in the name of the group in the experiment and the name of the 
        variable used to create the color gradient"""
        if show == 'all':
            show = self.data[name].keys()
        fig = px.scatter(self.data[name], x="Latitude", y="Longitude", color=gradient, title='Orchard Layout: ' + name, hover_data=show)
        fig.update_layout({'height':650, 'width':650})
        fig.update_xaxes({'fixedrange':True})
        #fig.update_yaxes({'fixedrange':True})
        fig.show()

class interactivePlot:
    def __init__(self, experiment, show='all'):
        self.experiment = experiment
        self.x_options = list(experiment.node.network.keys())
        self.y_options = self.x_options.copy()
        if show != 'all':
                for i in self.x_options.copy():
                        if i not in show:
                                self.x_options.remove(i)
                                self.y_options.remove(i)
        self.x_options.sort()
        self.y_options.sort()
        self.y_options += ['None (Distributions Only)']
        self.textbox1 = wd.Dropdown(
                description='x-Axis Variable: ',
                value=self.x_options[0],
                options=self.x_options
        )
        self.textbox2 = wd.Dropdown(
                description='y-Axis Variable: ',
                value=self.y_options[0],
                options=self.y_options
        )
        self.button = wd.RadioButtons(
                options=list(experiment.data.keys()) + ['All'],
                layout={'width': 'max-content'},
                description='Group',
                disabled=False
        )
        self.observe()
        self.showTrace()

    def display(self):
        container = wd.HBox([self.textbox1, self.textbox2])
        display(wd.VBox([container, self.g]))
        display(self.button)
        display(Nothing(), display_id='1')
        self.button.layout.display = 'none'

    def showTrace(self):
        traces = []
        for group in self.experiment.group_names:
            traces += [self.construct_trace(self.x_options[0], self.y_options[0], self.choose_trace(self.x_options[0], self.y_options[0]))(x=self.experiment.data[group][self.x_options[0]], y=self.experiment.data[group][self.y_options[0]], name=group)]
        go_layout = go.Layout(title=dict(text=self.x_options[0] + " vs. " + self.y_options[0]), barmode='overlay', height=500, width=800, xaxis=dict(title=self.x_options[0]), yaxis=dict(title=self.y_options[0]))
        self.g = go.FigureWidget(data=traces, layout=go_layout)

    def observe(self):
        self.textbox1.observe(self.response, names="value")
        self.textbox2.observe(self.response, names="value")
        self.button.observe(self.update_table, names='value')

    def choose_trace(self, x, y):
        xType, yType = self.experiment.node.nodeDict()[x].vartype, self.experiment.node.nodeDict()[y].vartype
        if xType != 'categorical' and yType != 'categorical':
            return 'scatter'
        elif xType == 'categorical' and yType != 'categorical':
            return 'bar'
        elif xType != 'categorical' and yType == 'categorical':
            return 'barh'
        else:
            return 'table'

    def construct_trace(self, x, y, traceType):
        if traceType == 'scatter':
            return lambda x={}, y={}, name=None: go.Scatter(x=x, y=y, mode='markers', opacity=0.75, name=name)
        elif traceType == 'bar':
            avg = self.experiment.data.groupby(x).agg('mean')
            std = self.experiment.data.groupby(x).agg('std')[y]
            return lambda x={}, y={}, name=None: go.Bar(x=list(avg.index), y=avg[y], name=name, error_y=dict(type='data', array=std[y]))
        elif traceType == 'barh':
            avg = self.experiment.data.groupby(y).agg('mean')
            std = self.experiment.data.groupby(y).agg('std')[x]
            return lambda x={}, y={}, name=None: go.Bar(x=avg[x], y=list(avg.index), name=name, error_y=dict(type='data', array=std[x]), orientation='h')
        elif traceType == 'table':
            return lambda x={}, y={}, name=None: go.Scatter()

    def pivot_table(self):
        if self.textbox1.value == self.textbox2.value:
            df = "Cannot create a pivot table with only one variable"
            return df
        if self.button.value == 'All':
            for group in self.experiment.group_names:
                df = pd.DataFrame()
                df = pd.concat([df, self.experiment.data[group]])
            df = df.groupby([self.textbox1.value, self.textbox2.value]).agg('count').reset_index().pivot(self.textbox1.value, self.textbox2.value, self.options[0])
        else:
            df = self.experiment.data[self.button.value].groupby([self.textbox1.value, self.textbox2.value]).agg('count').reset_index().pivot(self.textbox1.value, self.textbox2.value, self.options[0])
        return df

    def update_table(self, change):
        update_display(self.pivot_table(), display_id='1');
        self.button.layout.display = 'flex'

    def validate(self):
        return self.textbox1.value in self.x_options and self.textbox2.value in (self.x_options + ['None (Distributions Only)'])

    def response(self, change):
        if self.validate():
            if self.textbox2.value in self.x_options:
                traceType = self.choose_trace(self.textbox1.value, self.textbox2.value)
                with self.g.batch_update():
                    if traceType == 'table':
                        self.g.update_layout({'height':10, 'width':10})
                        self.g.layout.xaxis.title = ""
                        self.g.layout.yaxis.title = ""
                        self.g.layout.title = ""
                        self.button.layout.display = 'flex'
                    else:
                        if traceType == 'scatter':
                            for i in range(len(self.experiment.group_names)):
                                self.g.data[i].x = self.experiment.data[self.experiment.group_names[i]][self.textbox1.value]
                                self.g.data[i].y = self.experiment.data[self.experiment.group_names[i]][self.textbox2.value]
                                self.g.data[i].error_y = {'visible':False}
                                self.g.data[i].error_x = {'visible':False}
                                self.g.data[i].orientation = None
                            self.g.plotly_restyle({'type':'scatter', 'opacity':0.75})
                        elif traceType == 'bar':
                            self.g.plotly_restyle({'type':'bar', 'opacity':1})
                            for i in range(len(self.experiment.group_names)):
                                avg = self.experiment.data[self.experiment.group_names[i]].groupby(self.textbox1.value).agg('mean')
                                std = self.experiment.data[self.experiment.group_names[i]].groupby(self.textbox1.value).agg('std')[self.textbox2.value]
                                self.g.data[i].x = list(avg.index)
                                self.g.data[i].y = avg[self.textbox2.value]
                                self.g.data[i].error_y = {'type':'data', 'array':std, 'visible':True}
                                self.g.data[i].error_x = {'visible':False}
                                self.g.data[i].orientation = None
                        elif traceType == 'barh':
                            self.g.plotly_restyle({'type':'bar', 'opacity':1})
                            for i in range(len(self.experiment.group_names)):
                                avg = self.experiment.data[self.experiment.group_names[i]].groupby(self.textbox2.value).agg('mean')
                                std = self.experiment.data[self.experiment.group_names[i]].groupby(self.textbox2.value).agg('std')[self.textbox1.value]
                                self.g.data[i].x = avg[self.textbox1.value]
                                self.g.data[i].y = list(avg.index)
                                self.g.data[i].error_x = {'type':'data', 'array':std, 'visible':True}
                                self.g.data[i].orientation = 'h'
                                self.g.data[i].error_y  = {'visible':False}
                        self.g.layout.xaxis.title = self.textbox1.value
                        self.g.layout.yaxis.title = self.textbox2.value
                        self.g.layout.title = self.textbox1.value + " vs. " + self.textbox2.value
                        self.g.update_layout({'height':500, 'width':800})
                        update_display(Nothing(), display_id='1')
                        self.button.layout.display = 'none'
            else:
                with self.g.batch_update():
                    if self.experiment.node.nodeDict()[self.textbox1.value].vartype == "categorical":
                        self.g.plotly_restyle({'opacity':1})
                    else:
                        self.g.plotly_restyle({'opacity':0.75})
                    for i in range(len(self.experiment.group_names)):
                        self.g.data[i].x = self.experiment.data[self.experiment.group_names[i]][self.textbox1.value]
                        self.g.data[i].y = None
                        self.g.data[i].error_x = {'visible':False}
                        self.g.data[i].error_y = {'visible':False}
                        self.g.data[i].orientation = None
                    self.g.layout.xaxis.title = self.textbox1.value
                    self.g.layout.yaxis.title = "Count"
                    self.g.layout.title = self.textbox1.value
                    self.g.plotly_restyle({'type':'histogram'})

class Nothing:
    def __init__(self):
        None
    def __repr__(self):
        return ""

class CausalNetwork:
    def __init__(self, root_node):
        self.root_node = root_node

    def drawNetwork(self):
        return self.root_node.drawNetwork()

    def generate(self, config, runs):
        '''
        Performs experiment many times (runs) according to config, returns data
        '''
        self.data = dict()
        for c in config:
            self.data[c['name']] = [self.root_node.generate(c['N'], intervene=c['intervene']) for i in range(runs)]

    def statsContinuous(self, group, varx, vary):
        '''
        Calculates distribution of Pearson r and p-value between varx and vary (names of variables)
        '''
        runs = len(self.data[group])
        results = np.zeros((runs, 2))
        for i in range(runs):
            x = self.data[group][i][varx]
            y = self.data[group][i][vary]
            results[i] = sp.pearsonr(x, y)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(vary + ' vs. ' + varx + ', ' + str(runs) + ' runs')
        ax[0].hist(results[:,0])
        ax[0].set_title('Pearson r')
        ax[1].hist(np.log(results[:,1]))
        ax[1].set_title('log(p)')

    def statsAB(self, group0, group1, var):
        '''
        Calculates distribution of Welch's t and p-value of var between the null hypothesis (group0) and intervention (group1)
        '''
        runs = len(self.data[group0])
        results = np.zeros((runs, 2))
        for i in range(runs):
            a = self.data[group0][i][var]
            b = self.data[group1][i][var]
            results[i] = sp.ttest_ind(a, b, equal_var=True)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(var + ' between groups ' + group0 + ' and ' + group1 + ', ' + str(runs) + ' runs')
        ax[0].hist(results[:,0])
        ax[0].set_title("Welch's t")
        ax[1].hist(np.log(results[:,1]))
        ax[1].set_title('log(p)')

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
        nonlocal weights
        if weights is None:
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
    # Applies linear function on the input of func(*args[0:-1]), where the slope and intercept are determined by args[-1] according to x1, m1, c1, x2, m2, c2
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
    def f(x): # x is input value used to calculate mean and std of new distribution
        mean = M_mean[0]*x + c_mean
        std = max(M_std[0]*x + c_std, 0)
        return abs(np.random.normal(mean, std))
    return f

def categoricalLin(data): # data: {'category': (m, c, fuzz), etc}
    def f(x, y): # y is the category, x is the input value
        fuzz = data[y][2] if len(data[y]) == 3 else 0
        return data[y][0] * x + data[y][1] + np.random.normal(0, fuzz)
    return f

def categoricalGaussian(data): # data: {'category': (mean, std), etc}
    def f(x): # x is the category
        return np.random.normal(data[x][0], data[y][1])
    return f

'''
truffula
'''
# Uniformly distributed from 0m to 1000m
latitude_node = CausalNode('continuous', choice(np.linspace(0, 1000, 50), replace=False), name='Latitude', min=0, max=1000)
longitude_node = CausalNode('continuous', choice(np.linspace(0, 1000, 50), replace=False), name='Longitude', min=0, max=1000)
# Gaussian+absolute value, more wind in south
wind_node = CausalNode('continuous', lambda x,y: dependentGaussian(0, 2, 5, 1000, 10, 10)(x) + dependentGaussian(0, 6, 3, 1000, 2, 4)(x), name='Wind Speed', causes=[latitude_node, longitude_node], min=0, max=40)
supplement_node = CausalNode('categorical', constant('Water'), name='Supplement', categories=['Water', 'Kombucha', 'Milk', 'Tea'])
fertilizer_node = CausalNode('continuous', gaussian(10, 2), 'Fertilizer', min=0, max=20)
supplement_soil_effects = {'Water': (1, 0), 'Kombucha': (0.6, -5), 'Milk': (1.2, 10), 'Tea': (0.7, 0)}
# Fertilizer improves soil, kombucha destroys it
soil_node = CausalNode('continuous', lambda x, y: categoricalLin(supplement_soil_effects)(linear(0, 10, 20, 100, fuzz=5)(x), y), 'Soil Quality', causes=[fertilizer_node, supplement_node], min=0, max=100)
supplement_bees_effects = {'Water': (1, 0), 'Kombucha': (1.5, 0), 'Milk': (1, 0), 'Tea': (1.3, 0)}
# Beehive in north, bees avoid wind, love kombucha
bees_node = CausalNode('discrete', lambda x, y, z: categoricalLin(supplement_bees_effects)(dependentPoisson((0, 0, 250), (500, 30, 10), (0, 30, 40))(x, y), z), name='Number of Bees', causes=[latitude_node, wind_node, supplement_node], min=0, max=300)
# Bees and good soil improve fruiting
fruit_node = CausalNode('discrete', dependentPoisson((0, 0, 0), (100, 200, 28), (100, 50, 16)), name='Number of Fruits', causes=[soil_node, bees_node])
# fruit_node.drawNetwork()

truffula = CausalNetwork(fruit_node)

'''
basketball
'''
shottype_node = CausalNode('categorical', choice(['Above head', 'Layup', 'Hook shot'], weights=[6, 3, 2]), name='Shot Type', categories=['Above head', 'Layup', 'Hook shot'])
hours_node = CausalNode('continuous', choice(np.linspace(0, 14, 30), weights=1/np.linspace(1, 15, 30)), name='Hours Practised per Week')
height_node = CausalNode('continuous', gaussian(170, 10), name='Height (cm)', min=150, max=190)
ability_node = CausalNode('continuous', linearFunc(0, 1, 0, 10, 1, 20, linear(150, 40, 190, 60, fuzz=10)), name='Ability', causes=[height_node, hours_node])
shottype_modifier = {'Above head': (1, 0), 'Layup': (0.6, 0), 'Hook shot': (0.3, 0)}
success_node = CausalNode('continuous', categoricalLin(shottype_modifier), name='Success Rate', causes=[ability_node, shottype_node])

basketball = CausalNetwork(success_node)
