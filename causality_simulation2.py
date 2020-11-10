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

# TODO: Look at how Experiment parses init_data and migrate that to CausalNetwork
# Remove 'randomise' checkbox

display(HTML('''<style>
    [title="Assigned samples:"] { min-width: 150px; }
</style>'''))

def dialog(title, body, button):
    display(Javascript("require(['base/js/dialog'], function(dialog) {dialog.modal({title: '%s', body: '%s', buttons: {'%s': {}}})});" % (title, body, button)))

class CausalNode:
    def __init__(self, vartype, func, name, causes=None, min=0, max=100, categories=[], init=False):
        '''
        name: string, must be unique
        vartype: 'categorical', 'discrete', 'continuous'
        causes: (node1, ..., nodeN)
        func: f
        f is a function of N variables, matching the number of nodes and their types, returns a single number matching the type of this node
        self.network: {node_name: node, ...}, all nodes that the current node depends on
        init: True/False, whether variable is an initial immutable attribute
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
        self.init = init

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
                fix_all[name] = np.linspace(args[1], args[2], n)
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

class CausalNetwork:
    def __init__(self, root_node):
        self.root_node = root_node
        self.init_attr = [name for name, node in self.root_node.network.items() if node.init] # List of immutable attributes

    def drawNetwork(self):
        return self.root_node.drawNetwork()

    def generate(self, init_data, config, runs):
        '''
        Performs experiment many times (runs) according to config, returns data [i][group][var]
        config: dict {'name': group_name, 'samples_str': '1-100', 'intervention': {...}}
        '''
        self.data = []
        for i in range(runs):
            exp = Experiment(self, init_data)
            is_random = ''.join([g['samples_str'] for g in config]) == ''
            samples = randomAssign(exp.N, len(config)) if is_random else [text2Array(g['samples_str']) for g in config]
            groups = [{'name': config[i]['name'], 'samples': samples[i]} for i in range(len(config))]
            exp.setAssignment(groups)
            exp.doExperiment(config)
            self.data.append(exp.data)

    def statsContinuous(self, group, varx, vary):
        '''
        Calculates distribution of Pearson r and p-value between varx and vary (names of variables)
        '''
        runs = len(self.data)
        results = np.zeros((runs, 2))
        for i in range(runs):
            x = self.data[i][group][varx]
            y = self.data[i][group][vary]
            results[i] = sp.pearsonr(x, y)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(vary + ' vs. ' + varx + ', ' + str(runs) + ' runs')
        ax[0].hist(results[:,0])
        ax[0].set_title('Pearson r')
        ax[1].hist(np.log(results[:,1]))
        ax[1].set_title('log(p)')

    # def statsAB(self, group0, group1, var):
    #     '''
    #     Calculates distribution of Welch's t and p-value of var between the null hypothesis (group0) and intervention (group1)
    #     '''
    #     runs = len(self.data)
    #     results = np.zeros((runs, 2))
    #     for i in range(runs):
    #         a = self.data[i][group0][var]
    #         b = self.data[i][group1][var]
    #         results[i] = sp.ttest_ind(a, b, equal_var=False)
    #     fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    #     fig.suptitle(var + ' between groups ' + group0 + ' and ' + group1 + ', ' + str(runs) + ' runs')
    #     ax[0].hist(results[:,0])
    #     ax[0].set_title("Welch's t")
    #     ax[1].hist(np.log(results[:,1]))
    #     ax[1].set_title('log(p)')

    def statsAB(self, group0, group1, var, resamples=1000):
        '''
        Permutation test
        '''
        runs = len(self.data)
        results = np.zeros((runs, 2))
        for i in range(runs):
            a = self.data[i][group0][var]
            b = self.data[i][group1][var]
            na = len(a)
            nb = len(b)
            sample_all = np.concatenate((a, b))
            results[i,0] = abs(np.mean(a) - np.mean(b))
            mean_diffs = np.zeros(resamples)
            for j in range(resamples):
                permuted = np.random.permutation(sample_all)
                mean_diffs[j] = abs(np.mean(permuted[0:na]) - np.mean(permuted[na+1:]))
            results[i,1] = np.sum(mean_diffs>=results[i,0]) / resamples
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(var + ' between groups ' + group0 + ' and ' + group1 + ', ' + str(runs) + ' runs')
        ax[0].hist(results[:,0])
        ax[0].set_title("Difference in mean")
        ax[1].hist(np.log(results[:,1]))
        ax[1].set_title('log(p)') # p = probability that a random assignment into A, B groups will give (abs) mean greater than the observed one

class Experiment:
    def __init__(self, network, init_data):
        '''
        init_data: dict of name, array to initialise basic immutable attributes. Keys must match init_attr in instance of Network
        '''
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
        self.p = None

    def assignment(self, config=None, hide_random=False):
        '''
        UI for group assignment of samples
        config: list of dicts, each being {'name': group_name, 'samples_str': string}
        samples_str: e.g. '1-25,30,31-34', if all groups have empty string '' then assume randomise
        '''
        self.group_assignment = groupAssignment(self)
        if config is not None:
            self.group_assignment.setAssignment(config, hide_random)
            self.submitAssignment()

    def setAssignment(self, groups):
        '''
        Sets assignment into self.groups without UI
        groups: list of dicts, each being {'name': group_name, samples: [array]}
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
        # Populate self.data for plotOrchard
        for g in self.groups:
            mask = [i in g['samples'] for i in range(self.N)]
            d = dict()
            for node_name, arr in self.init_data.items():
                d[node_name] = arr[mask] 
            d['id'] = np.array(g['samples'])+1
            self.data[g['name']] = pd.DataFrame(d)
        if self.p:
            self.p.updateAssignments()
        else:
            self.plotAssignment()

    def plotAssignment(self):
        '''
        Can be implemented differently in different scenarios
        '''
        self.p = assignmentPlot(self)

    def setting(self, show='all', config=None, disable=[]):
        '''
        Let user design experiment
        disabled: array of names
        show: array of names
        '''
        disable = self.node.network if disable == 'all' else disable
        self.intervention_setting = interventionSetting(self, show=show, disable=disable)
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
        if msg:
            display(wd.Label(value='Data from experiment collected!'))

    def plot(self, show='all'):
        p = interactivePlot(self, show)
        self.p = p
        p.display()

    def plotOrchard(self, gradient=None, show='all'):
        '''
        Takes in the name of the group in the experiment and the name of the 
        variable used to create the color gradient
        '''
        o = orchardPlot(self, gradient=gradient, show=show)
        self.o = o
        o.display()

class groupAssignment:
    def __init__(self, experiment):
        '''
        UI for group assignment of samples
        submitAssignment: callback function
        '''
        self.experiment = experiment
        wd.Label(value='Sample size: %d' % self.experiment.N)
        self.randomise_button = wd.Button(description='Randomise assignment', layout=wd.Layout(width='180px'))
        self.group_assignments = [singleGroupAssignment(1)]
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
        self.group_assignments.append(singleGroupAssignment(i+1))
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

class singleGroupAssignment:
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

class interventionSetting:
    def __init__(self, experiment, show='all', disable=[]):
        self.experiment = experiment
        self.group_settings = [singleGroupInterventionSetting(self.experiment, g, show=show, disable=disable) for g in self.experiment.groups]
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

class singleGroupInterventionSetting:
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
        self.node_settings = [singleNodeInterventionSetting(self.experiment.node.network[name], disable=name in disable) for name in to_list]

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

class singleNodeInterventionSetting:
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

class assignmentPlot:
    def __init__(self, experiment):
        self.experiment = experiment
        self.group_names = experiment.group_names
        self.data = experiment.data
        self.buildTraces()
        self.layout = go.Layout(title=dict(text='Tree Group Assignments'),barmode='overlay', height=650, width=800,
                              xaxis=dict(title='Longitude', fixedrange=True), yaxis=dict(title='Latitude', fixedrange=True),
                              hovermode='closest',
                              margin=dict(b=80, r=200, autoexpand=False),
                              showlegend=True)
        self.plot = go.FigureWidget(data=self.traces, layout=self.layout)
        display(self.plot)
        
    def buildTraces(self):
        self.traces = []
        self.group_names = self.experiment.group_names
        self.data = self.experiment.data
        for i, name in enumerate(self.group_names):
            self.traces += [go.Scatter(x=self.data[name]['Longitude'], y=self.data[name]['Latitude'], mode='markers', hovertemplate='Latitude: %{x} <br>Longitude: %{y} <br>', marker_symbol=i, name=name)]
        
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
        elif traceType == 'scatter':
            for group in self.experiment.group_names:
                data = self.experiment.data[group]
                traces += [go.Scatter(x=data[x], y=data[y], mode='markers', opacity=0.75, name=group)]
                annotations += [dict(xref='paper',yref='paper',x=0.5, y=annotation_y, showarrow=False, text=self.display_values(group))]
                annotation_y += -0.05
        elif traceType == 'bar':
            for group in self.experiment.group_names:
                avg = self.experiment.data.groupby(x).agg('mean')
                std = self.experiment.data.groupby(x).agg('std')[y]
                traces += [go.Bar(x=list(avg.index), y=avg[y], name=group, error_y=dict(type='data', array=std[y]))]
                annotations += [dict(xref='paper',yref='paper',x=0.5, y=annotation_y, showarrow=False, text=self.display_values(group))]
                annotation_y += -0.05
        elif traceType == 'barh':
            for group in self.experiment.group_names:
                avg = self.experiment.data.groupby(y).agg('mean')
                std = self.experiment.data.groupby(y).agg('std')[x]
                traces += [go.Bar(x=avg[x], y=list(avg.index), name=group, error_y=dict(type='data', array=std[x]), orientation='h')]
                annotations += [dict(xref='paper',yref='paper',x=0.5, y=annotation_y, showarrow=False, text=self.display_values(group))]
                annotation_y += -0.05
        go_layout = go.Layout(title=dict(text=x if traceType == 'histogram' else x + " vs. " + y ),
                              barmode='overlay',
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
latitude_node = CausalNode('continuous', choice(np.linspace(0, 1000, 50), replace=False), name='Latitude', min=0, max=1000, init=True)
longitude_node = CausalNode('continuous', choice(np.linspace(0, 1000, 50), replace=False), name='Longitude', min=0, max=1000, init=True)
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
height_node = CausalNode('continuous', gaussian(170, 10), name='Height (cm)', min=150, max=190, init=True)
ability_node = CausalNode('continuous', linearFunc(0, 1, 0, 10, 1, 20, linear(150, 40, 190, 60, fuzz=10)), name='Ability', causes=[height_node, hours_node])
shottype_modifier = {'Above head': (1, 0), 'Layup': (0.6, 0), 'Hook shot': (0.3, 0)}
success_node = CausalNode('continuous', categoricalLin(shottype_modifier), name='Success Rate', causes=[ability_node, shottype_node])

basketball = CausalNetwork(success_node)