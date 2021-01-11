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

'''
Data structure
--------------
CausalNetwork
* complete network of all nodes
* init_data
* no per-experiment data

CausalNode
* causal relations from a single node

Experiment
* group assignment
* intervention
'''

# TODO: assignment(config=) change format config to {'group_name': '1-500'} (but maybe a list of dicts is better for preserving ordering)
# TODO: assignment string must be within the total number of samples
# TODO: assignment mustn't have overlapping groups

display(HTML('''<style>
    [title="Assigned samples:"] { min-width: 150px; }
</style>'''))

def dialog(title, body, button):
    display(Javascript("require(['base/js/dialog'], function(dialog) {dialog.modal({title: '%s', body: '%s', buttons: {'%s': {}}})});" % (title, body, button)))

class CausalNetwork:
    def __init__(self):
        self.nodes = {} # 'node_name': node
        self.init_data = pd.DataFrame()

    def addNode(self, node):
        if node.name in self.nodes.keys(): # check for duplicate nodes
            raise ValueError('A node with the same name %s already exists!' % node['name'])
        self.nodes[node.name] = node

    def addCause(self, effect, cause):
        self.nodes[effect].setCause(func=lambda x: x, causes=[cause]) # func has to be an unlifted function, so don't use identity here!

    # to be run at the beginning of each notebook
    def init(self, data):
        '''
        initialises the network and its constituent nodes
        data: {'some_node': [val, val, ...], ...} all arrays must have same size
        '''
        l = []
        for name, n in self.nodes.items():
            if n.causes is None: # is an init node, populate init_data column, throws error if arrays don't have same size
                self.init_data[name] = data[name]
            n.root_causes = n.traceRoots()
            self.replacePlaceholders(n)                
        # TODO: validate causal network has no single-direction loops (other loops are allowed)

    def generateSingle(self, row):
        '''
        Returns one row of pandas table
        row: single row of DataFrame or dict {'some_node': val, ...}
        '''
        total_nodes = len(self.nodes)
        new_row = {name: row[name] for name in self.nodes.keys() if name in row.keys() and not pd.isnull(row[name])} # populate new_row with values of init nodes or fixed nodes
        while len(new_row) < total_nodes:
            for name, n in self.nodes.items():
                if name not in new_row.keys():
                    ready = True # checks if all its causes have been evaluated or fixed
                    for c in n.root_causes:
                        if c not in new_row.keys():
                            ready = False
                            break
                    if ready: # evaluate this node using values from its .causes
                        cause_vals = {c: new_row[c] for c in n.root_causes}
                        new_row[name] = n.evaluate(cause_vals)
        return new_row

    # replaces all PlaceholderNodes of a node's causes by the actual node searched by name
    def replacePlaceholders(self, node):
        if isinstance(node, PlaceholderNode):
            raise ValueError('A PlaceholderNode cannot be passed directly to replacePlaceholders. Use it on the parent instead.')
        elif node.causes is not None:
            for i, n in enumerate(node.causes):
                if isinstance(n, PlaceholderNode):
                    if n.name not in self.nodes:
                        raise ValueError("Node %s doesn't exist in the causal network!" % n.name)
                    node.causes[i] = self.nodes[n.name] # replace
                else:
                    self.replacePlaceholders(n) # recurse down the tree

    def draw(self, root, intermediate=False):
        '''
        Returns a graphviz visualization of the underlying causal network.
        root: the root node in the visualization
        intermediate: set True to show unnamed intermediate nodes for debugging
        '''
        g = Digraph(name=root)
        def draw_edges(node, g):
            if node.causes:
                causes = node.causes if intermediate else [self.nodes[name] for name in node.root_causes]
                for c in causes:
                    g.edge(str(c.name), str(node.name))
                    draw_edges(c, g)
        draw_edges(self.nodes[root], g)
        return g

class CausalNode:
    def __init__(self, name=None):
        self.name = name if name else id(self) # given name or use unique id
        self.causes = None # array of node type arguments to f. None means this is an init node
        self.func = None # only takes positional arguments
        self.root_causes = set() # set of names of PlaceholderNodes that this node directly depends on

    def setCause(self, func, causes):
        self.causes = causes
        self.func = func

    def traceRoots(self):
        '''
        Stores a list of names of (string-named) nodes that this node depends on
        '''
        root_causes = set()
        if self.causes is None:
            return root_causes
        for c in self.causes:
            if isinstance(c, PlaceholderNode):
                root_causes.add(c.name)
            else:
                root_causes = root_causes.union(c.traceRoots()) # recurse down the network
        return root_causes

    def evaluate(self, params):
        '''
        Evaluates the value of this node given the values of nodes in root_causes
        kwargs.keys() must match root_causes TODO: validate this somewhere?
        '''
        if self.causes is None:
            raise ValueError('This is an init node. evaluate() should not have been called.')
        cause_vals = [None] * len(self.causes)
        for i, c in enumerate(self.causes):
            if isinstance(c.name, str): # if predefined node (i.e. not intermediate node)
                cause_vals[i] = params[c.name]
            else: # recurse down the tree
                cause_vals[i] = c.evaluate(params)
        return self.func(*cause_vals)

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

class Experiment:
    def __init__(self, network):
        self.network = network
        self.N = len(self.network.init_data)
        self.data = self.network.init_data.copy()
        self.assigned = False # has the assignment step been run
        self.done = False # has the experiment been done
        self.a = None # AssignmentPlot, overwritten when .plotAssignment is run

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
        Populates the 'Group' column of self.data without UI
        groups: list of dicts, each being {'name': group_name, 'samples': [array]}
        '''
        # check for duplicates and populate self.data
        self.data['Group'] = None # eventually the column should all be group names
        seen = []
        for g in groups:
            if g['name'] in seen:
                dialog('Duplicate group names', 'Some of the groups have been given the same name. Please choose a unique name for each group.', 'OK')
                return
            else:
                seen.append(g['name'])
            if g['samples'] is None:
                dialog('Assignment invalid', 'Invalid assignment of samples to groups! Please revise your assignments.', 'OK')
                return
            for i in g['samples']:
                self.data.at[i,'Group'] = g['name']
        self.groups = [g['name'] for g in groups] # list of group names
        self.group_ids = {name: i for i, name in enumerate(self.groups)} # for easier lookup of group ids
        if None in self.data['Group'].unique():
            dialog('Not all samples assigned', 'Not all samples have been assigned to a group! Please revise your assignments.', 'OK')
            return
        self.assigned = True
        if self.a:
            self.a.updateAssignments()
        else:
            self.plotAssignment()

    def submitAssignment(self, sender=None):
        '''
        Collects the group assignments from UI
        Checks for duplicate group names
        '''
        self.setAssignment(self.group_assignment.getAssignment())

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
        disable = self.network.nodes.keys() if disable == 'all' else disable
        self.intervention_setting = InterventionSetting(self, show=show, disable=disable)
        if config is not None:
            self.intervention_setting.setIntervention(config)
            self.doExperiment(config)

    def generate(self, intervention=[]):
        '''
        Generates experimental data for all groups according to intervention. Stores results in .data
        intervention: list of dicts, each being {'name': 'some_group', 'intervention': {'some_node': [args], ...}, ...}
        intervention format:
        ['fixed', val] (val could be number or name of category)
        ['range', start, end]
        ['array', [...]] array size must be n
        '''
        # reset the dataframe to just group assignments
        for key in self.data:
            if key != 'Group':
                self.data.drop(key, axis=1)
        for g in intervention:
            group = g['name']
            inter = g['intervention']
            n = len(self.data.loc[self.data['Group'] == group].index) # number of samples in group
            for name, args in inter.items():
                if args[0] == 'fixed':
                    to_fix = args[1]
                elif args[0] == 'range':
                    to_fix = np.random.permutation(np.linspace(args[1], args[2], n))
                    if isinstance(self, DiscreteNode):
                        to_fix = np.rint(to_fix)
                elif args[0] == 'array':
                    to_fix = args[1]
                # populate only those entries that are intervened on
                self.data.loc[self.data['Group'] == group, name] = to_fix
        # generate the rest of each row using the existing values
        for i in range(len(self.data)):
            new_vals = self.network.generateSingle(self.data.loc[i])
            for name, val in new_vals.items():
                self.data.at[i, name] = val # use loc or iloc?

    def doExperiment(self, intervention, msg=False):
        '''
        Perform experiment under intervention
        intervention: {'group_name': {'node_name': [...]}}
        '''
        self.generate(intervention)
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
        p = InteractivePlot(self, show)
        self.p = p
        p.display()

class TruffulaExperiment(Experiment):
    def __init__(self):
        super().__init__(truffula)

    def plotAssignment(self):
        self.a = AssignmentPlot(self)

    def plotOrchard(self, gradient=None, show='all'):
        '''
        Plots a scatterplot of the orchard layout with coloring to represent the variables listed in SHOW.
        The default coloring is from the variable listed under GRADIENT, but there is a dropdown list for students to select
        another variable to visualize.
        '''
        if not self.done:
            dialog('Experiment not performed', 'You have not yet performed the experiment! Click on "Perform experiment" before running this box.', 'OK')
            return
        o = OrchardPlot(self, gradient=gradient, show=show)
        self.o = o
        o.display()

class BasketballExperiment(Experiment):
    def __init__(self):
        super().__init__(basketball)

    def plotAssignment(self):
        self.a = AssignmentPlot(self)

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
        self.group_settings = [SingleGroupInterventionSetting(self.experiment, g, len(self.experiment.data.loc[self.experiment.data['Group'] == g].index), show=show, disable=disable) for g in self.experiment.groups]
        submit = wd.Button(description='Perform experiment')
        display(submit)
        submit.on_click(self.submit)

    def submit(self, sender=None):
        self.experiment.doExperiment(self.getIntervention(), msg=True)

    def getIntervention(self):
        return [{'name': s.name, 'intervention': s.getIntervention()} for s in self.group_settings]

    def setIntervention(self, config):
        for c in config:
            j = self.experiment.group_ids[c['name']]
            self.group_settings[j].setIntervention(c)

class SingleGroupInterventionSetting:
    def __init__(self, experiment, name, N, show='all', disable=[]):
        '''
        UI settings for a single group
        config: {'name': group_name, 'samples': [sample_ids]}
        '''
        self.experiment = experiment
        self.name = name
        self.N = N
        group_text = wd.Label(value='Group name: %s, %d samples' % (name, N))
        display(group_text)
        to_list = list(self.experiment.network.nodes.keys()) if show == 'all' else show
        to_list = [a for a in to_list if a not in self.experiment.network.init_data.keys()]
        to_list.sort()
        self.node_settings = [SingleNodeInterventionSetting(self.experiment.network.nodes[n], disable=n in disable) for n in to_list]

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
        self.is_categorical = isinstance(node, CategoricalNode)
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
        a = 0 if self.is_categorical else node.min
        b = 0 if self.is_categorical else node.max
        self.range_arg1 = wd.BoundedFloatText(min=a, max=b, disabled=True, layout=wd.Layout(width='70px', visibility=self.range_visibility))
        self.range_arg2_text = wd.Label(value='to', layout=wd.Layout(visibility=self.range_visibility, width='15px'))
        self.range_arg2 = wd.BoundedFloatText(min=a, max=b, disabled=True, layout=wd.Layout(width='70px', visibility=self.range_visibility))
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

class AssignmentPlot:
    '''
    AssignmentPlots are displayed when a student clicks "Show Assignment" to select a way to assign variables to
    groups--they depict the student's choice of group assignments. This plot must be specified for each type of experiment
    because different experiments have different initial attributes that require different kinds of visualizations.
    e.g. the Basketball experiment must plot students with different heights, the Truffula experiment must plot trees and
    their x, y coordinates
    '''
    def __init__(self, experiment):
        self.experiment = experiment
        self.group_names = experiment.groups
        self.data = experiment.data
        self.buildTraces()
        if type(self.experiment).__name__ == 'TruffulaExperiment':
            self.layout = go.Layout(title=dict(text='Tree Group Assignments'), barmode='overlay', height=650, width=800,
                                  xaxis=dict(title='Longitude', fixedrange=True), yaxis=dict(title='Latitude', fixedrange=True),
                                  hovermode='closest',
                                  margin=dict(b=80, r=200, autoexpand=False),
                                  showlegend=True)
        elif type(self.experiment).__name__ == 'BasketballExperiment':
            self.layout = go.Layout(title=dict(text='Student Group Assignments'), barmode='overlay', height=650, width=800,
                                  xaxis=dict(title='Student', fixedrange=True),
                                  yaxis=dict(title='Height (cm)', fixedrange=True, range=(120, 200)),
                                  hovermode='closest',
                                  margin=dict(b=80, r=200, autoexpand=False),
                                  showlegend=True)
        else: # must be extended for other kinds of experiments, layouts go here
            pass
        self.plot = go.FigureWidget(data=self.traces, layout=self.layout)
        display(self.plot)
        
    def buildTraces(self):
        '''
        Builds the plotly.go traces that are to be included in the visualization. Differs by type of experiment PLOT,
        so extend here if adding another experiment.
        '''
        self.traces = []
        self.group_names = self.experiment.groups
        self.data = self.experiment.data
        if type(self.experiment).__name__ == 'TruffulaExperiment':
            for i, name in enumerate(self.group_names):
                self.traces += [go.Scatter(x=self.data[self.data['Group'] == name]['Longitude'], y=self.data[self.data['Group'] == name]['Latitude'], mode='markers', hovertemplate='Latitude: %{x} <br>Longitude: %{y} <br>', marker_symbol=i, name=name)]
        elif type(self.experiment).__name__ == 'BasketballExperiment':
            for i, name in enumerate(self.group_names):
                self.traces += [go.Bar(x=self.data[self.data['Group'] == name].index, y=self.data[self.data['Group'] == name]['Height (cm)'], hovertemplate='Student: %{x} <br>Height: %{y} cm<br>', name=name)]
        else: # must be extended for other kinds of experiments, traces go here
            pass
        
    def updateAssignments(self):
        '''
        Updates the assignment plot's traces and layout when student clicks Visualise Assignment after having already visualized
        the assignment once
        '''
        self.buildTraces()
        with self.plot.batch_update():
            self.plot.data = []
            for trace in self.traces:
                self.plot.add_traces(trace)
            self.plot.layout = self.layout

class OrchardPlot:
    '''
    For TruffulaExperiment only; plots a scatterplot of the trees' coordinate locations, colored by the variable GRADIENT
    and with a dropdown list that includes all variables listed in SHOW. Includes a colorbar.
    '''
    def __init__(self, experiment, gradient=None, show='all'):
        self.data = experiment.data
        self.experiment = experiment
        if show == 'all':
            self.options = [col for col in self.data.columns if type(col) is str and col != 'Group']
        else:
            self.options = show
        for node in experiment.network.nodes:
            if isinstance(node, CategoricalNode) and node.name in show:
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
                for i, name in enumerate(self.experiment.groups):
                    self.g.data[i].marker.color = self.data[self.data['Group'] == name][self.textbox.value]
                    self.g.update_layout({'coloraxis':{'colorscale':'Plasma', 'colorbar':{'title':self.textbox.value}}})
                    self.g.data[i].hovertemplate = 'Latitude: %{x} <br>Longitude: %{y} <br>' + self.textbox.value + ': %{marker.color}<br>'

    def plotOrchard(self, gradient):
        '''
        Creates/Initializes a FigureWidget object with a scatterplot of the trees and colors it with the variable GRADIENT
        '''
        traces = []
        for i, name in enumerate(self.experiment.groups):
            traces += [go.Scatter(x=self.data[self.data['Group'] == name]['Longitude'], y=self.data[self.data['Group'] == name]['Latitude'],
                                 marker=dict(color=self.data[self.data['Group'] == name][gradient], coloraxis='coloraxis'),
                                 mode='markers',
                                 name=name,
                                 hovertemplate='Latitude: %{x} <br>Longitude: %{y} <br>'+ self.textbox.value + ': %{marker.color}<br>', hoverlabel=dict(namelength=0), marker_symbol=i)]
        width = 700 if (len(self.experiment.groups) == 1) else 725 + max([len(name) for name in self.experiment.groups])*6.5
        go_layout = go.Layout(title=dict(text='Orchard Layout'),barmode='overlay', height=650, width=width,
                              xaxis=dict(title='Longitude', fixedrange=True, range=[-50, 1050]), 
                              yaxis=dict(title='Latitude', fixedrange=True, range=[-50, 1050]),
                              hovermode='closest', legend=dict(yanchor="top", y=1, xanchor="left", x=1.25),
                              coloraxis={'colorscale':'Plasma', 'colorbar':{'title':gradient}})
        self.g = go.FigureWidget(data=traces, layout=go_layout)
        
    def display(self):
        container = wd.HBox([self.textbox])
        display(wd.VBox([container, self.g]))

class InteractivePlot:
    '''
    An interactive plot that plots experiment results. Includes a dropdown list that changes the variables plotted. Automatically
    chooses the plot type depending on the type of variables involved.
    '''
    def __init__(self, experiment, show='all'):
        self.experiment = experiment
        self.x_options = list(experiment.network.nodes.keys())
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
        '''
        Displays the correlations and p-values for each group in an annotation below the plot
        '''
        text = ""
        xType, yType = type(self.experiment.network.nodes[self.textbox1.value]).__name__, type(self.experiment.network.nodes[self.textbox2.value]).__name__
        if xType != 'CategoricalNode' and yType != 'CategoricalNode':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                x = self.experiment.data[self.experiment.data['Group'] == group][self.textbox1.value]
                y = self.experiment.data[self.experiment.data['Group'] == group][self.textbox2.value]
                r, p, ci_lo, ci_hi = pearsonr_ci(x,y)
            text += group + ': ' + 'Correlation (r) is ' + '{0:#.3f}, '.format(r) + 'P-value is ' + '{0:#.3g}, '.format(p) +f'confidence interval for corr is [{round(ci_lo,2)}, {round(ci_hi,2)}] '
        return text

    def createTraces(self, x, y):
        '''
        Returns the traces and layout depending on the appropriate trace type for the variables selected (X and Y)
        '''
        traces = []
        annotations = []
        annotation_y = -0.20 - 0.02*len(self.experiment.groups)
        traceType = self.choose_trace(x, y)
        if traceType == 'histogram':
            for group in self.experiment.groups:
                data = self.experiment.data[self.experiment.data['Group'] == group]
                if type(self.experiment.network.nodes[x]).__name__ == 'CategoricalNode': # if counts of categories, make opacity 1 because bars will not overlap
                    opacity = 1
                else:
                    opacity = 0.75
                traces += [go.Histogram(x=data[x], name=group, bingroup=1, opacity=opacity)]
                y = 'Count'
                barmode = 'overlay'
        elif traceType == 'scatter':
            for group in self.experiment.groups:
                data = self.experiment.data[self.experiment.data['Group'] == group]
                traces += [go.Scatter(x=data[x], y=data[y], mode='markers', opacity=0.75, name=group)]
                annotations += [dict(xref='paper',yref='paper',x=0.5, y=annotation_y, showarrow=False, text=self.display_values(group))] # adds annotations with spacing that include the correlation/p-value
                annotation_y += -0.05
                barmode = 'overlay'
        elif traceType == 'bar':
            for group in self.experiment.groups:
                data = self.experiment.data[self.experiment.data['Group'] == group]
                avg = data.groupby(x).agg('mean')
                std = data.groupby(x).agg('std')[y]
                traces += [go.Bar(x=list(avg.index), y=avg[y], name=group, error_y=dict(type='data', array=std))]
                annotations += [dict(xref='paper',yref='paper',x=0.5, y=annotation_y, showarrow=False, text=self.display_values(group))] # adds annotations with spacing that include the correlation/p-value
                annotation_y += -0.05
                barmode = 'group'
        elif traceType == 'barh':
            for group in self.experiment.groups:
                data = self.experiment.data[self.experiment.data['Group'] == group]
                avg = data.groupby(y).agg('mean')
                std = data.groupby(y).agg('std')[x]
                traces += [go.Bar(x=avg[x], y=list(avg.index), name=group, error_x=dict(type='data', array=std), orientation='h')]
                annotations += [dict(xref='paper',yref='paper',x=0.5, y=annotation_y, showarrow=False, text=self.display_values(group))] # adds annotations with spacing that include the correlation/p-value
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
        '''
        Initializes a FigureWidget with the traces and layout
        '''
        traces, layout = self.createTraces(self.x_options[0], self.y_options[0])
        self.g = go.FigureWidget(layout=layout)
        for t in traces:
            self.g.add_traces(t)
            
    def updateTraces(self):
        '''
        Updates the FigureWidget with the appropriate traces and layout after a new dropdown item is selected
        '''
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
        '''
        Returns the appropriate trace type given the variables X and Y according to their variable types
        '''
        if y == 'None (Distributions Only)':
            return 'histogram'
        xType, yType = type(self.experiment.network.nodes[x]).__name__, type(self.experiment.network.nodes[y]).__name__
        if xType != 'CategoricalNode' and yType != 'CategoricalNode':
            return 'scatter'
        elif xType == 'CategoricalNode' and yType != 'CategoricalNode':
            return 'bar'
        elif xType != 'CategoricalNode' and yType == 'CategoricalNode':
            return 'barh'
        else:
            return 'table'

    def pivot_table(self):
        '''
        Creates a pivot table for categorical variables
        '''
        if self.textbox1.value == self.textbox2.value:
            df = "Cannot create a pivot table with only one variable"
            return df
        if self.button.value == 'All':
            for group in self.experiment.groups:
                df = pd.DataFrame()
                df = pd.concat([df, self.experiment.data[group]])
            df = df.groupby([self.textbox1.value, self.textbox2.value]).agg('count').reset_index().pivot(self.textbox1.value, self.textbox2.value, self.options[0])
        else:
            df = self.experiment.data[self.button.value].groupby([self.textbox1.value, self.textbox2.value]).agg('count').reset_index().pivot(self.textbox1.value, self.textbox2.value, self.options[0])
        return df

    def update_table(self, change):
        '''
        Displays the pivot table and button display for pivot table options if applicable
        '''
        update_display(self.pivot_table(), display_id='1');
        self.button.layout.display = 'flex'

    def validate(self):
        return self.textbox1.value in self.x_options and self.textbox2.value in (self.x_options + ['None (Distributions Only)'])

    def response(self, change):
        '''
        Updates the display when a new dropdown item is selected
        '''
        if self.validate():
            traceType = self.choose_trace(self.textbox1.value, self.textbox2.value)
            with self.g.batch_update():
                if traceType == 'table': # if the variables are both categorical, displays a pivot table
                    self.g.update_layout({'height':10, 'width':10})
                    self.g.layout.xaxis.title = ""
                    self.g.layout.yaxis.title = ""
                    self.g.layout.title = ""
                    self.button.layout.display = 'flex'
                else:
                    self.updateTraces()
                    update_display(Nothing(), display_id='1') # update pivot table display with nothing if not categorical
                    self.button.layout.display = 'none'

class Nothing:
    '''
    Used with IPython.display (particularly when updating displays) to display nothing
    '''
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
            fargs = [(fnargs[node_inds_lookup[i]] if isinstance(v, CausalNode) else args[i]) for i, v in enumerate(args)] # replaces the nodes in args by the corresponding values in fnargs
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
    if floor > ceil:
        raise ValueError('floor has to be less than or equal to ceil.')
    if x<floor:
        return floor
    elif x<ceil:
        return x
    else:
        return ceil

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
    '''
    any number of arguments, m is either a number or an array of the same size as args
    points: array of n points, each with n entries, where n = len(args)+1, from which m and c can be solved
    '''
    if points:
        m, c = solveLinear(points)
    elif len(args) == 1:
        m = [m]
    return np.array(m)@np.array(args)+c

@lift
def categoricalLinear(x, category, params):
    m, c = params[category]
    return m*x+c

@lift
def choice(opts, weights=None, replace=True):
    if weights is None:
        chosen = np.random.choice(opts, replace=replace)
    else:
        weights = np.array(weights)
        p = weights/np.sum(weights)
        chosen = np.random.choice(opts, p=p, replace=replace)
    return chosen

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

def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = sp.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = sp.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

'''
truffula
'''
truffula = CausalNetwork()
truffula.addNode(ContinuousNode(name='Latitude', min=0, max=1000))
truffula.addNode(ContinuousNode(name='Longitude', min=0, max=1000))
truffula.addNode(ContinuousNode(name='Wind speed', min=0, max=40))
truffula.addCause(effect='Wind speed',
    cause=bound(
        normal(
            mean=linear(
                dist(node('Latitude'), node('Longitude'), point=(3000, 2000)), # absolute distance from a special point
                points=[(2236, 20), (3605, 3)] # mean speed=20 if close to point, =3 if far from point
                ),
            stdev=linear(
                dist(node('Latitude'), node('Longitude'), point=(1010, 800)),
                points=[(2236, 10), (3605, 3)] # stdev=10 if close to point, =3 if far from point
                )
            ),
        floor=0)
    )
truffula.addNode(CategoricalNode(name='Supplement', categories=['Water', 'Kombucha', 'Milk', 'Tea']))
truffula.addCause(effect='Supplement',
    cause=identity('Water')
    )
truffula.addNode(ContinuousNode(name='Fertilizer', min=0, max=20))
truffula.addCause(effect='Fertilizer',
    cause=normal(mean=10, stdev=2)
    )
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
                bound(
                    linear(
                        dist(node('Latitude'), node('Longitude'), point=(-50, 30)), # distance from bee hive
                        node('Wind speed'),
                        points=[(30, 0, 250), (1000, 30, 10), (30, 30, 40)] # rate=250 if close to hive with no wind, =10 if far from hive with high wind, =40 if close to hive with high wind
                        ),
                    floor=0
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
        bound(
            linear(
                node('Soil quality'), node('Number of bees'),
                points=[(0, 0, 0), (100, 200, 28), (100, 50, 16)] # rate=0 if poor soil and no bees, =28 if good soil and high bees, =16 if good soil and some bees
                ),
            floor=0
            )
        )
    )

'''
basketball
'''
basketball = CausalNetwork()
basketball.addNode(CategoricalNode(name='Shot type', categories=['Above head', 'Layup', 'Hook shot']))
basketball.addCause(effect='Shot type',
    cause=choice(opts=['Above head', 'Layup', 'Hook shot'], weights=[6, 3, 2])
    )
basketball.addNode(ContinuousNode(name='Hours practised per week', min=0, max=20))
basketball.addCause(effect='Hours practised per week',
    cause=choice(opts=np.linspace(0, 14, 30), weights=1/np.linspace(1, 15, 30))
    )
basketball.addNode(ContinuousNode(name='Height (cm)', min=140, max=200))
basketball.addNode(ContinuousNode(name='Ability', min=0, max=100))
basketball.addCause(effect='Ability',
    cause=linear(
        linear(
            node('Height (cm)'),
            points=[(150, 40), (190, 60)] # 40 if height=150, 60 if height=190
            ),
        m=1,
        c=linear(
            node('Hours practised per week'),
            points=[(0, 0), (10, 20)] # add 0 if hours=0, add 20 if hours=10
            )
        )
    )
basketball.addNode(ContinuousNode(name='Success rate', min=0, max=100))
basketball.addCause(effect='Success rate',
    cause=bound(
        categoricalLinear(
            node('Ability'),
            category=node('Shot type'),
            params={'Above head': (1, 0), 'Layup': (0.6, 0), 'Hook shot': (0.3, 0)}
            ),
        floor=0,
        ceil=100
        )
    )