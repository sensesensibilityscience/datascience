import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

class Test:
    def __init__(self, config):
        '''
        config: {
        'mean_neg': ,
        'mean_pos': ,
        'std_neg': ,
        'std_pos': ,
        'p_pos': 
        }
        '''
        self.mean_neg = config['mean_neg']
        self.mean_pos = config['mean_pos']
        self.std_neg = config['std_neg']
        self.std_pos = config['std_pos']
        self.p_pos = config['p_pos']
        self.title = config['title']
        
    def singleTest(self):
        '''
        Generates a single row [test_result_from_normal_distribution, known_1_or_0]
        '''
        if np.random.rand() < self.p_pos:
            return [np.random.normal(self.mean_pos, self.std_pos), 1]
        else:
            return [np.random.normal(self.mean_neg, self.std_neg), 0]
        
    def generate(self, n):
        '''
        Generates table of test results
        '''
        self.results = np.array([self.singleTest() for i in range(n)])
    
    def countDiagnoses(self, threshold):
        '''
        Returns number of true/false pos/neg based on known test results and threshold
        threshold: above is positive, below is negative
        '''
        true_pos = sum((self.results[:,0] > threshold) * (self.results[:,1] == 1))
        true_neg = sum((self.results[:,0] <= threshold) * (self.results[:,1] == 0))
        false_pos = sum((self.results[:,0] > threshold) * (self.results[:,1] == 0))
        false_neg = sum((self.results[:,0] <= threshold) * (self.results[:,1] == 1))
        return true_pos, true_neg, false_pos, false_neg
    
    def plotDiagnoses(self, threshold):
        fig, ax = plt.subplots(figsize=(6,4))
        bins = np.arange(0, 360, 5)
        counts_neg, bins, patches = ax.hist(self.results[self.results[:,1]==0,0], bins=bins, fc=(31/255,119/255,180/255,.7), label='Tested Negative')
        counts_pos, bins, patches = ax.hist(self.results[self.results[:,1]==1,0], bins=bins, fc=(255/255,127/255,14/255,0.7), label='Tested Positive')
        counts = np.concatenate((counts_neg, counts_pos))
        ax.plot([threshold, threshold], [0, max(counts+10)], c='black')
        ax.set_xlim(min(self.results[:,0])-20, max(self.results[:,0])+20)
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
        
    def plot(self):
        threshold = (min(self.results[:,0]) + max(self.results[:,0]))/2
        fig, ax = self.plotDiagnoses(threshold)
        @widgets.interact(t=(0, 350, 5))
        def update(t):
            ax.lines[0].set_xdata([t, t])
            self.printDiagnoses(t)