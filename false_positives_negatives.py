import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
#hi winston
class Test:
    def __init__(self, config, data=None):
        '''
        config: {
        'mean_neg': ,
        'mean_pos': ,
        'std_neg': ,
        'std_pos': ,
        'p_pos': ,
        'label_neg': ,
        'label_pos':
        }
        data is a nx3 array of numbers. [:,0] is test number, [:,1] column is count of positives [:,2] column is count of negatives
        '''
        if data is not None:
            self.data = data
            self.title = config['title']
            self.label_neg = config['label_neg']
            self.label_pos = config['label_pos']
        else:
            self.data = None
            self.mean_neg = config['mean_neg']
            self.mean_pos = config['mean_pos']
            self.std_neg = config['std_neg']
            self.std_pos = config['std_pos']
            self.p_pos = config['p_pos']
            self.title = config['title']
            self.label_neg = config['label_neg']
            self.label_pos = config['label_pos']
            self.floor = config['floor'] if 'floor' in config.keys() else None
            self.ceil = config['ceil'] if 'ceil' in config.keys() else None
        #JUST TESTING
    def singleTest(self):
        '''
        Generates a single row [test_result_from_normal_distribution, known_1_or_0]
        '''
        if np.random.rand() < self.p_pos:
            mean = self.mean_pos
            std = self.std_pos
            posneg = 1
        else:
            mean = self.mean_neg
            std = self.std_neg
            posneg = 0
        sample = np.random.normal(mean, std)
        if self.floor != None:
            sample = max(self.floor, sample)
        if self.ceil != None:
            sample = min(self.ceil, sample)
        return [sample, posneg]

    def generate(self, n):
        '''
        Generates table of test results
        '''
        if self.data is None: # From normal distribution
            self.results = np.array([self.singleTest() for i in range(n)])
        else: # From data
            total = sum(self.data[:,1]) + sum(self.data[:,2])
            pool = []
            weights = []
            for row in self.data:
                pool.append([row[0], 1])
                pool.append([row[0], 0])
                weights.append(row[1]/total)
                weights.append(row[2]/total)
            ids = np.random.choice(range(len(pool)), size=n, p=weights)
            self.results = np.array([pool[i] for i in ids])

    def countDiagnoses(self, threshold):
        '''
        Returns number of true/false pos/neg based on known test results and threshold
        threshold: above is positive, below is negative
        '''
        true_pos = sum((self.results[:,0] >= threshold) * (self.results[:,1] == 1))
        true_neg = sum((self.results[:,0] < threshold) * (self.results[:,1] == 0))
        false_pos = sum((self.results[:,0] >= threshold) * (self.results[:,1] == 0))
        false_neg = sum((self.results[:,0] < threshold) * (self.results[:,1] == 1))
        return true_pos, true_neg, false_pos, false_neg
    
    def plotDiagnoses(self, threshold, stepsize=None):
        tmin = min(self.results[:,0])
        tmax = max(self.results[:,0])
        stepsize = (tmax-tmin)/100 if stepsize is None else stepsize
        tmin -= 5*stepsize
        tmax += 5*stepsize
        bins = np.arange(tmin, tmax, stepsize)
        fig, ax = plt.subplots(figsize=(6,4))
        counts_neg, bins, patches = ax.hist(self.results[self.results[:,1]==0,0], bins=bins, fc=(31/255,119/255,180/255,.7), label=self.label_neg)
        counts_pos, bins, patches = ax.hist(self.results[self.results[:,1]==1,0], bins=bins, fc=(255/255,127/255,14/255,0.7), label=self.label_pos)
        counts = np.concatenate((counts_neg, counts_pos))
        ax.plot([threshold, threshold], [0, max(counts*1.05)], c='black')
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(0, max(counts+10))
        ax.set_xlabel('Test Value')
        ax.set_ylabel('Count')
        ax.set_title(self.title)
        ax.legend()
        return fig, ax
    
    def printDiagnoses(self, threshold):
        true_pos, true_neg, false_pos, false_neg = self.countDiagnoses(threshold)
        print('True positives: %d' % true_pos)
        print('True negatives: %d' % true_neg)
        print('False positives: %d' % false_pos)
        print('False negatives: %d' % false_neg)
        print('False positive rates: {:.2f}%'.format(false_pos/(true_pos+false_pos)*100))
        print('False negative rates: {:.2f}%'.format(false_neg/(true_neg+false_neg)*100))
        print('Overall accuracy: {:.2f}%'.format((true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)*100))
        
    def plot(self, stepsize=None):
        tmin = min(self.results[:,0])
        tmax = max(self.results[:,0])
        stepsize = (tmax-tmin)/100 if stepsize is None else stepsize
        threshold = (tmin+tmax)/2
        tmin -= 5*stepsize
        tmax += 5*stepsize
        fig, ax = self.plotDiagnoses(threshold)
        @widgets.interact(t=(tmin, tmax, stepsize))
        def update(t):
            ax.lines[0].set_xdata([t, t])
            self.printDiagnoses(t)