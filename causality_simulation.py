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

# TODO: use data frame, maybe extra column for group assignment (dtype=string)
# Experiment.data = {'Number of Bees': [...], 2109384712098347: [...]} (actually a dataframe)

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
        for name, n in self.nodes.items():
            self.replacePlaceholders(n)
        l = [] # collect lengths
        for name in self.init_data.keys(): # populate init_data with data, raises error if any key in init_data is not present in data
            self.init_data[name] = data[name]
            l.append(len(data[name]))
        self.N = l[0] # sample size
        if min(l) != max(l):
            raise ValueError('Every array in init_data must have the same length.')
        # TODO: validate causal network has no single-direction loops (other loops are allowed)
        # validate init_data matches with the nodes that have .causes == None

    # TODO: is this even necessary??
    def setRoot(self, node):
        self.root_node = node

    # replaces all PlaceholderNodes of a node's causes by the actual node searched by name
    def replacePlaceholders(self, node):
        if isinstance(node, PlaceholderNode):
            raise ValueError('A PlaceholderNode cannot be passed directly to replacePlaceholders. Use it on the parent instead.')
        elif node.causes != None:
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
        self.network = network
        self.init_data = self.network.init_data
        self.N = self.network.N
        self.data = {} # {group_name: {node_name: [val, ...], ...}, ...}
        self.assigned = False # has the assignment step been run
        self.done = False # has the experiment been done

    def assignment(self, config=None, hide_random=False):
        '''
        UI for group assignment of samples
        config: list of dicts, each being {'name': group_name, 'samples_str': string}
        samples_str: e.g. '1-25,30,31-34', if all groups have empty string '' then assume randomise
        '''
        self.group_assignment = GroupAssignment(self)
        if config is not None:
            self.group_assignment.setAssignment(config, hide_random)
            self.submitAssignment()

    def setAssignment(self, groups):
        '''
        Sets assignment into self.groups without UI
        groups: list of dicts, each being {'name': group_name, 'samples': [array]}
        '''
        self.groups = groups
        seen = set()
        self.group_ids = dict()
        for i in range(len(self.groups)):
            name = self.groups[i]['name']
            if name not in seen:
                seen.add(name)
                self.group_ids[name] = i
            else:
                dialog('Duplicate group names', 'Some of the groups have been given the same name. Please choose a unique name for each group.', 'OK')
                return
        self.group_names = list(self.group_ids.keys())

    def submitAssignment(self, sender=None):
        '''
        Collects the group assignments from UI
        self.groups: list of dicts, each being {'name': group_name, samples: [array]}
        self.group_ids: dict {'group_name': id} for easier reverse lookup
        Checks for duplicate group names
        '''
        self.setAssignment(self.group_assignment.getAssignment())
        self.assigned = True
        # Populate self.data for plotOrchard
        for g in self.groups:
            mask = [i in g['samples'] for i in range(self.N)]
            d = dict()
            for node_name, arr in self.init_data.items():
                d[node_name] = arr[mask] 
            d['id'] = np.array(g['samples'])+1
            self.data[g['name']] = pd.DataFrame(d)
        self.plotAssignment()

    def plotAssignment(self):
        '''
        Can be implemented differently in different scenarios. Extend the Experiment class and override this method
        '''
        pass

    def setting(self, show='all', config=None, disable=[]):
        '''
        Let user design experiment
        disabled: array of names
        show: array of names
        '''
        if not self.assigned:
            dialog('Groups not assigned', 'You have not yet assigned any groups! Click on "Visualise assignment" before running this box.', 'OK')
            return
        disable = self.node.network if disable == 'all' else disable
        self.intervention_setting = InterventionSetting(self, show=show, disable=disable)
        if config is not None:
            self.intervention_setting.setIntervention(config)
            self.doExperiment(config)

    def doExperiment(self, intervention, msg=False):
        '''
        Perform experiment under intervention
        intervention: list of dictionaries, each being {'name': group_name, 'intervention': {'node_name', [...]}}
        '''
        self.data = dict()
        for g in intervention:
            j = self.group_ids[g['name']]
            mask = [i in self.groups[j]['samples'] for i in range(self.N)]
            for node_name, arr in self.init_data.items():
                g['intervention'][node_name] = ['array', arr[mask]]
            N_samples = len(self.groups[self.group_ids[g['name']]]['samples'])
            self.data[g['name']] = self.node.generate(N_samples, intervention=g['intervention'])
        self.done = True
        if msg:
            display(wd.Label(value='Data from experiment collected!'))

    def plot(self, show='all'):
        '''
        Plots data after doExperiment has been called
        '''
        if not self.done:
            dialog('Experiment not performed', 'You have not yet performed the experiment! Click on "Perform experiment" before running this box.', 'OK')
            return
        p = interactivePlot(self, show)
        self.p = p
        p.display()

class GroupAssignment:
    def __init__(self, experiment):
        '''
        UI for group assignment of samples
        submitAssignment: callback function
        '''
        self.experiment = experiment
        wd.Label(value='Sample size: %d' % self.experiment.N)
        self.randomise_button = wd.Button(description='Randomise assignment', layout=wd.Layout(width='180px'))
        self.group_assignments = [SingleGroupAssignment(1)]
        self.add_group_button = wd.Button(description='Add another group')
        self.submit_button = wd.Button(description='Visualise assignment')
        self.box = wd.VBox([g.box for g in self.group_assignments])
        display(self.randomise_button, self.box, self.add_group_button, self.submit_button)
        self.randomise_button.on_click(self.randomise)
        self.add_group_button.on_click(self.addGroup)
        self.submit_button.on_click(self.experiment.submitAssignment)

    def setAssignment(self, config, hide_random):
        for i in range(len(config)-1):
            self.addGroup()
        self.greyAll()
        for i in range(len(config)):
            self.group_assignments[i].setName(config[i]['name'])
        if ''.join([g['samples_str'] for g in config]) == '':
            self.randomise()
        else:
            for i in range(len(config)):
                self.group_assignments[i].setSamples(config[i]['samples_str'])
        if hide_random:
            self.randomise_button.layout.visibility = 'hidden'

    def addGroup(self, sender=None):
        i = self.group_assignments[-1].i
        self.group_assignments.append(SingleGroupAssignment(i+1))
        self.box.children = [g.box for g in self.group_assignments]

    def getAssignment(self):
        '''
        Reads the settings and returns a list of dictionaries
        '''
        return [g.getAssignment() for g in self.group_assignments]

    def randomise(self, sender=None):
        '''
        Randomly assigns samples to groups and changes settings in UI
        '''
        N = self.experiment.N
        N_group = len(self.group_assignments)
        assigned_ids = randomAssign(N, N_group)
        for i in range(N_group):
            self.group_assignments[i].samples.value = array2Text(assigned_ids[i])

    def greyAll(self):
        self.randomise_button.disabled = True
        self.add_group_button.disabled = True
        self.submit_button.disabled = True
        for g in self.group_assignments:
            g.greyAll()

class SingleGroupAssignment:
    def __init__(self, i):
        '''
        UI for a single line of group assignment
        '''
        self.i = i # Group number
        self.name = 'Group %d' % i
        i_text = wd.Label(value=self.name, layout=wd.Layout(width='70px'))
        self.group_name = wd.Text(description='Name:')
        self.samples = wd.Text(description='Assigned samples:', layout=wd.Layout(width='400px'))
        self.box = wd.HBox([i_text, self.group_name, self.samples])

    def getAssignment(self):
        '''
        Returns dict {'name': group_name, 'samples': [list_of_sample_ids]}
        '''
        assignment = dict()
        self.name = self.name if self.group_name.value == '' else self.group_name.value
        assignment['name'] = self.name
        assignment['samples'] = text2Array(self.samples.value)
        return assignment

    def setName(self, name):
        self.group_name.value = name

    def setSamples(self, samples):
        self.samples.value = samples

    def greyAll(self):
        self.group_name.disabled = True
        self.samples.disabled = True

class InterventionSetting:
    def __init__(self, experiment, show='all', disable=[]):
        self.experiment = experiment
        self.group_settings = [SingleGroupInterventionSetting(self.experiment, g, show=show, disable=disable) for g in self.experiment.groups]
        submit = wd.Button(description='Perform experiment')
        display(submit)
        submit.on_click(self.submit)

    def submit(self, sender=None):
        self.experiment.doExperiment(self.getIntervention(), msg=True)

    def getIntervention(self):
        return [{'name': s.name, 'N': s.N, 'intervention': s.getIntervention()} for s in self.group_settings]

    def setIntervention(self, config):
        for c in config:
            j = self.experiment.group_ids[c['name']]
            self.group_settings[j].setIntervention(c)

class SingleGroupInterventionSetting:
    def __init__(self, experiment, config, show='all', disable=[]):
        '''
        UI settings for a single group
        config: {'name': group_name, 'samples': [sample_ids]}
        '''
        self.experiment = experiment
        self.name = config['name']
        self.N = len(config['samples'])
        group_text = wd.Label(value='Group name: %s, %d samples' % (self.name, self.N))
        display(group_text)
        to_list = list(self.experiment.node.network.keys()) if show == 'all' else show
        to_list.sort()
        self.node_settings = [SingleNodeInterventionSetting(self.experiment.node.network[name], disable=name in disable) for name in to_list]

    def getIntervention(self):
        intervention = dict()
        for s in self.node_settings:
            inter = s.getIntervention()
            if inter is not None:
                intervention[s.name] = inter
        return intervention

    def setIntervention(self, config):
        for s in self.node_settings:
            if s.name in config['intervention'].keys():
                s.setIntervention(config['intervention'][s.name])

class SingleNodeInterventionSetting:
    def __init__(self, node, disable=False):
        '''
        Single line of radio buttons and text boxes for intervening on a single variable in a single group
        '''
        self.name = node.name
        self.disable = disable
        self.is_categorical = node.vartype == 'categorical'
        self.indent = wd.Label(value='', layout=wd.Layout(width='20px'))
        self.text = wd.Label(value=self.name, layout=wd.Layout(width='180px'))
        self.none = wd.RadioButtons(options=['No intervention'], layout=wd.Layout(width='150px'))
        self.fixed = wd.RadioButtons(options=['Fixed'], layout=wd.Layout(width='70px'))
        self.fixed.index = None
        if self.is_categorical:
            fixed_arg = wd.Dropdown(options=node.categories, disabled=True, layout=wd.Layout(width='100px'))
        else:
            fixed_arg = wd.BoundedFloatText(disabled=True, layout=wd.Layout(width='70px'))
        self.fixed_arg = fixed_arg
        self.range_visibility = 'hidden' if self.is_categorical else 'visible'
        self.range = wd.RadioButtons(options=['Range'], layout=wd.Layout(width='70px', visibility=self.range_visibility))
        self.range.index = None
        self.range_arg1_text = wd.Label(value='from', layout=wd.Layout(visibility=self.range_visibility, width='30px'))
        self.range_arg1 = wd.BoundedFloatText(min=node.min, max=node.max, disabled=True, layout=wd.Layout(width='70px', visibility=self.range_visibility))
        self.range_arg2_text = wd.Label(value='to', layout=wd.Layout(visibility=self.range_visibility, width='15px'))
        self.range_arg2 = wd.BoundedFloatText(min=node.min, max=node.max, disabled=True, layout=wd.Layout(width='70px', visibility=self.range_visibility))
        self.none.observe(self.none_observer, names=['value'])
        self.fixed.observe(self.fixed_observer, names=['value'])
        self.range.observe(self.range_observer, names=['value'])
        self.box = wd.HBox([self.indent, self.text, self.none, self.fixed, self.fixed_arg, self.range, self.range_arg1_text, self.range_arg1, self.range_arg2_text, self.range_arg2])
        display(self.box)
        if self.disable:
            self.greyAll()

    def greyAll(self):
        self.none.disabled = True
        self.fixed.disabled = True
        self.fixed_arg.disabled = True
        self.range.disabled = True
        self.range_arg1.disabled = True
        self.range_arg2.disabled = True

    def setIntervention(self, intervention):
        if intervention[0] == 'fixed':
            self.fixed.index = 0
            self.fixed_arg.value = intervention[1]
        elif intervention[0] == 'range':
            self.range.index = 0
            self.range_arg1.value = intervention[1]
            self.range_arg2.value = intervention[2]

    # Radio button .index = None if off, .index = 0 if on
    def none_observer(self, sender):
        if self.none.index == 0:
            self.fixed.index = None
            self.fixed_arg.disabled = True
            self.range.index = None
            self.range_arg1.disabled = True
            self.range_arg2.disabled = True
        if self.disable:
            self.greyAll()

    def fixed_observer(self, sender):
        if self.fixed.index == 0:
            self.none.index = None
            self.fixed_arg.disabled = False
            self.range.index = None
            self.range_arg1.disabled = True
            self.range_arg2.disabled = True
        if self.disable:
            self.greyAll()

    def range_observer(self, sender):
        if self.range.index == 0:
            self.none.index = None
            self.fixed.index = None
            self.fixed_arg.disabled = True
            self.range_arg1.disabled = False
            self.range_arg2.disabled = False
        if self.disable:
            self.greyAll()

    def getIntervention(self):
        '''
        generates intervention from UI settings
        '''
        if self.none.index == 0: # None is deselected, 0 is selected
            return None
        elif self.fixed.index == 0:
            return ['fixed', self.fixed_arg.value]
        elif self.range.index == 0:
            return ['range', self.range_arg1.value, self.range_arg2.value]

class TruffulaExperiment(Experiment):
    def __init__(self, network):
        super().__init__(network)
        self.p = None # plot

    def plotAssignment():
        pass

    def plot():
        pass

    def plotOrchard(self, gradient=None, show='all'):
        '''
        Takes in the name of the group in the experiment and the name of the 
        variable used to create the color gradient
        '''
        if not self.done:
            dialog('Experiment not performed', 'You have not yet performed the experiment! Click on "Perform experiment" before running this box.', 'OK')
            return
        o = orchardPlot(self, gradient=gradient, show=show)
        self.o = o
        o.display()

class BasketballExperiment(Experiment):
    def __init__(self, network):
        super().__init__(network)
        self.p = None # plot

class assignmentPlot:
    def __init__(self, experiment, plot='Truffula'):
        self.experiment = experiment
        self.group_names = experiment.group_names
        self.data = experiment.data
        self.plot = plot
        self.buildTraces()
        if self.plot == 'Truffula':
            self.layout = go.Layout(title=dict(text='Tree Group Assignments'),barmode='overlay', height=650, width=800,
                                  xaxis=dict(title='Longitude', fixedrange=True), yaxis=dict(title='Latitude', fixedrange=True),
                                  hovermode='closest',
                                  margin=dict(b=80, r=200, autoexpand=False),
                                  showlegend=True)
        else:
            self.layout = go.Layout(title=dict(text='Student Group Assignments'),barmode='overlay', height=650, width=800,
                                  xaxis=dict(title='Student', fixedrange=True),
                                  yaxis=dict(title='Height', fixedrange=True, range=(120, 200)),
                                  hovermode='closest',
                                  margin=dict(b=80, r=200, autoexpand=False),
                                  showlegend=True)
        self.plot = go.FigureWidget(data=self.traces, layout=self.layout)
        display(self.plot)
        
    def buildTraces(self):
        self.traces = []
        self.group_names = self.experiment.group_names
        self.data = self.experiment.data
        if self.plot == 'Truffula':
            for i, name in enumerate(self.group_names):
                self.traces += [go.Scatter(x=self.data[name]['Longitude'], y=self.data[name]['Latitude'], mode='markers', hovertemplate='Latitude: %{x} <br>Longitude: %{y} <br>', marker_symbol=i, name=name)]
        else:
            for i, name in enumerate(self.group_names):
                self.traces += [go.Bar(x=self.data[name]['id'], y=self.data[name]['Height (cm)'], hovertemplate='Student: %{x} <br>Height: %{y} cm<br>', name=name)]
        
    def updateAssignments(self):
        self.buildTraces()
        with self.plot.batch_update():
            self.plot.data = []
            for trace in self.traces:
                self.plot.add_traces(trace)
            self.plot.layout = self.layout

class orchardPlot:
    def __init__(self, experiment, gradient=None, show='all'):
        self.data = experiment.data
        self.experiment = experiment
        self.options = self.data[experiment.group_names[0]].columns.tolist()
        if show != 'all':
            for i in self.options.copy():
                if i not in show:
                    self.options.remove(i)
        for name in experiment.node.nodeDict():
            if experiment.node.nodeDict()[name].vartype == 'categorical' and name in show:
                self.options.remove(name)
        self.options.sort()
        if not gradient:
            gradient = self.options[0]
        self.textbox = wd.Dropdown(
                description='Gradient: ',
                value=gradient,
                options=self.options
        )
        self.textbox.observe(self.response, names="value")
        self.plotOrchard(gradient)
        
    def validate(self):
        return self.textbox.value in self.options

    def response(self, change):
        if self.validate():
            with self.g.batch_update():
                for i, name in enumerate(self.experiment.group_names):
                    self.g.data[i].marker.color = self.data[name][self.textbox.value]
                    self.g.update_layout({'coloraxis':{'colorscale':'Plasma', 'colorbar':{'title':self.textbox.value}}})
                    self.g.data[i].hovertemplate = 'Latitude: %{x} <br>Longitude: %{y} <br>' + self.textbox.value + ': %{marker.color}<br>'

    def plotOrchard(self, gradient):
        """Takes in the name of the group in the experiment and the name of the 
        variable used to create the color gradient"""
        traces = []
        for i, name in enumerate(self.experiment.group_names):
            traces += [go.Scatter(x=self.data[name]['Latitude'], y=self.data[name]['Longitude'],
                                 marker=dict(color=self.data[name][gradient], coloraxis='coloraxis'),
                                 mode='markers',
                                 name=name,
                                 hovertemplate='Latitude: %{x} <br>Longitude: %{y} <br>'+ self.textbox.value + ': %{marker.color}<br>', hoverlabel=dict(namelength=0), marker_symbol=i)]
        width = 700 if (len(self.experiment.group_names) == 1) else 725 + max([len(name) for name in self.experiment.group_names])*6.5
        go_layout = go.Layout(title=dict(text='Orchard Layout'),barmode='overlay', height=650, width=width,
                              xaxis=dict(title='Latitude', fixedrange=True, range=[-50, 1050]), 
                              yaxis=dict(title='Longitude', fixedrange=True, range=[-50, 1050]),
                              hovermode='closest', legend=dict(yanchor="top", y=1, xanchor="left", x=1.25),
                              coloraxis={'colorscale':'Plasma', 'colorbar':{'title':gradient}})
        self.g = go.FigureWidget(data=traces, layout=go_layout)
        
    def display(self):
        container = wd.HBox([self.textbox])
        display(wd.VBox([container, self.g]))

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
        self.initTraces()

    def display(self):
        container = wd.HBox([self.textbox1, self.textbox2])
        display(wd.VBox([container, self.g]))
        display(self.button)
        display(Nothing(), display_id='1')
        self.button.layout.display = 'none'
    
    def display_values(self, group):
        text = ""
        xType, yType = self.experiment.node.nodeDict()[self.textbox1.value].vartype, self.experiment.node.nodeDict()[self.textbox2.value].vartype
        if xType != 'categorical' and yType != 'categorical':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = sp.pearsonr(self.experiment.data[group][self.textbox1.value], self.experiment.data[group][self.textbox2.value])
            text += group + ': ' + 'Correlation (r) is ' + '{0:#.3f}, '.format(r[0]) + 'P-value is ' + '{0:#.3g}'.format(r[1])
        return text

    def createTraces(self, x, y):
        traces = []
        annotations = []
        annotation_y = -0.20 - 0.02*len(self.experiment.group_names)
        traceType = self.choose_trace(x, y)
        if traceType == 'histogram':
            for group in self.experiment.group_names:
                data = self.experiment.data[group]
                if self.experiment.node.nodeDict()[x].vartype == 'categorical':
                    opacity = 1
                else:
                    opacity = 0.75
                traces += [go.Histogram(x=data[x], name=group, bingroup=1, opacity=opacity)]
                y = 'Count'
                barmode = 'overlay'
        elif traceType == 'scatter':
            for group in self.experiment.group_names:
                data = self.experiment.data[group]
                traces += [go.Scatter(x=data[x], y=data[y], mode='markers', opacity=0.75, name=group)]
                annotations += [dict(xref='paper',yref='paper',x=0.5, y=annotation_y, showarrow=False, text=self.display_values(group))]
                annotation_y += -0.05
                barmode = 'overlay'
        elif traceType == 'bar':
            for group in self.experiment.group_names:
                avg = self.experiment.data[group].groupby(x).agg('mean')
                std = self.experiment.data[group].groupby(x).agg('std')[y]
                traces += [go.Bar(x=list(avg.index), y=avg[y], name=group, error_y=dict(type='data', array=std))]
                annotations += [dict(xref='paper',yref='paper',x=0.5, y=annotation_y, showarrow=False, text=self.display_values(group))]
                annotation_y += -0.05
                barmode = 'group'
        elif traceType == 'barh':
            for group in self.experiment.group_names:
                avg = self.experiment.data[group].groupby(y).agg('mean')
                std = self.experiment.data[group].groupby(y).agg('std')[x]
                traces += [go.Bar(x=avg[x], y=list(avg.index), name=group, error_x=dict(type='data', array=std), orientation='h')]
                annotations += [dict(xref='paper',yref='paper',x=0.5, y=annotation_y, showarrow=False, text=self.display_values(group))]
                annotation_y += -0.05
                barmode = 'group'
        go_layout = go.Layout(title=dict(text=x if traceType == 'histogram' else x + " vs. " + y ),
                              barmode=barmode,
                              height=500+50,
                              width=800,
                              xaxis=dict(title=x), yaxis=dict(title=y),
                              annotations = annotations,
                              margin=dict(b=80+50, r=200, autoexpand=False))
        return traces, go_layout
    
    def initTraces(self):
        traces, layout = self.createTraces(self.x_options[0], self.y_options[0])
        self.g = go.FigureWidget(layout=layout)
        for t in traces:
            self.g.add_traces(t)
            
    def updateTraces(self):
        self.g.data = []
        traces, layout = self.createTraces(self.textbox1.value, self.textbox2.value)
        for t in traces:
            self.g.add_traces(t)
        self.g.layout.annotations = layout.annotations
        self.g.layout = layout

    def observe(self):
        self.textbox1.observe(self.response, names="value")
        self.textbox2.observe(self.response, names="value")
        self.button.observe(self.update_table, names='value')

    def choose_trace(self, x, y):
        if y == 'None (Distributions Only)':
            return 'histogram'
        xType, yType = self.experiment.node.nodeDict()[x].vartype, self.experiment.node.nodeDict()[y].vartype
        if xType != 'categorical' and yType != 'categorical':
            return 'scatter'
        elif xType == 'categorical' and yType != 'categorical':
            return 'bar'
        elif xType != 'categorical' and yType == 'categorical':
            return 'barh'
        else:
            return 'table'

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
            traceType = self.choose_trace(self.textbox1.value, self.textbox2.value)
            with self.g.batch_update():
                if traceType == 'table':
                    self.g.update_layout({'height':10, 'width':10})
                    self.g.layout.xaxis.title = ""
                    self.g.layout.yaxis.title = ""
                    self.g.layout.title = ""
                    self.button.layout.display = 'flex'
                else:
                    self.updateTraces()
                    update_display(Nothing(), display_id='1')
                    self.button.layout.display = 'none'

class Nothing:
    def __init__(self):
        None
    def __repr__(self):
        return ""

def text2Array(text):
    text = text.replace(' ', '')
    if re.fullmatch(r'^((\d+)(|-(\d+)),)*(\d+)(|-(\d+))$', text) is None:
        return None
    matches = re.findall(r'((\d+)-(\d+))|(\d+)', text)
    ids = []
    for m in matches:
        if m[3] != '':
            ids = np.concatenate((ids, [int(m[3])-1])) # Subtract one because text starts at 1, array starts at 0
        else:
            if int(m[2]) < int(m[1]):
                return None
            else:
                ids = np.concatenate((ids, np.arange(int(m[1])-1, int(m[2]))))
    uniq = list(set(ids))
    uniq.sort()
    if len(ids) != len(uniq):
        return None
    return uniq

def array2Text(ids):
    ids.sort()
    ids = np.array(ids)+1 # Add one because text starts at 1, array starts at 0
    segments = []
    start = ids[0]
    end = ids[0]
    for j in range(len(ids)):
        if j == len(ids)-1:
            end = ids[j]
            s = str(start) if start == end else '%d-%d' % (start, end)
            segments.append(s)
        elif ids[j+1] != ids[j]+1:
            end = ids[j]
            s = str(start) if start == end else '%d-%d' % (start, end)
            segments.append(s)
            start = ids[j+1]
    return ','.join(segments)

def randomAssign(N, N_group):
    '''
    Randomly assigns N total items into N_group groups
    Returns a list of lists of ids
    '''
    arr = np.arange(N)
    np.random.shuffle(arr)
    result = []
    for i in range(N_group):
        start = i*N//N_group
        end = min((i+1)*N//N_group, N)
        result.append(arr[start:end])
    return result

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
# truffula.setRoot(node='Number of fruits')