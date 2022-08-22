import spacy
import lemminflect
import ipywidgets as wd
from IPython.display import display, HTML, Javascript
import json

'''
TODO
* Fix apostrophes and hyphens having spaces
* Add square grid with colours graphic
* Change number entry into sliders
* Graphic and final number react in real time to the sliders
* Add option to display equations or not
'''

# load English language model (large size)
nlp = spacy.load('en_core_web_lg')

def getInfo(text):
    '''
    Print POS tagging, for debugging
    '''
    for w in nlp(text):
        print(w.text, w.lemma_, w.pos_, w.tag_)

def stripPunct(doc_dep):
    '''
    Remove punctuations
    '''
    return [w for w in doc_dep if w.tag_ != '.']

def getSubject(doc):
    '''
    Get subject of sentence
    
    TODO: Pick up 'there' as well!
    '''
    for token in doc:
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return start, end

def statementify(question, negate=False):
    '''
    Turn a question like 'did John eat the apple?' to a statement 'John ate the apple', respecting the tense.
    
    TODO: Turn 'I' questions into 'you' statements.
    '''
    try:
        doc_dep = stripPunct(nlp(question))
        verbs = [(i, w) for i, w in enumerate(doc_dep) if w.tag_ in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']]
        assert verbs[0][0] == 0
        if len(verbs) == 2 and verbs[0][1].lemma_ == 'do' and negate == False:
            verb_inflected = verbs[1][1]._.inflect(verbs[0][1].tag_)
            text_new = [w.text for w in doc_dep[1:verbs[1][0]]] + [verb_inflected] + [w.text for w in doc_dep[verbs[1][0]+1:]]
        else:
            start, end = getSubject(doc_dep)
            text_new = [w.text for w in doc_dep[1:end]] + [verbs[0][1].text.lower()]
            text_new += ['not'] if negate else []
            text_new += [w.text for w in doc_dep[end:]]
    except:
        return 'the answer to "%s" is no'%question if negate else 'the answer to "%s" is yes'%question
    return ' '.join(text_new)

def bayes(prior, true_pos_rate, false_pos_rate, pos):
    '''
    Calculate the updated posterior given a prior and true/false positive rates of the test.
    '''
    if pos:
        return prior * true_pos_rate / (prior * true_pos_rate + (1 - prior) * false_pos_rate)
    else:
        return prior * (1 - true_pos_rate) / (prior * (1 - true_pos_rate) + (1 - prior) * (1 - false_pos_rate))

def toJS(q1, q2):
    statement = statementify(q1)
    statement_neg = statementify(q1, negate=True)
    test = q2
    to_js = dict(statement=statement, statement_neg=statement_neg, test=test)
    with open('base_rate.json', 'w') as f:
        f.write(json.dumps(to_js))

def load():
    display(HTML('''<style>
    #questions {
        height: 80px;
    }

    #sliders {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
        width: 100%;
        height: 90px;
        text-align: center;
    }

    #slider1 {
        flex: 33.33%;
    }

    #slider2 {
        flex: 33.33%;
    }

    #slider3 {
        flex: 33.33%;
    }

    .slider text {
        font-size: 12pt;
    }

    .slider_label {
        font-size: 14pt;
    }

    #graphic {
        width: 100%;
        text-align: center;
        font-size: 12pt;
    }

    .legend_text {
        text-align: left;
    }

    #my_tooltip {
        position: absolute;
        width: 240px;
        opacity: 0;
        background-color: #386fb0;
        color: #fffffb;
        border-radius: 5px;
        padding: 12px;
        box-shadow: 3px 3px 2px #ddd;
        text-align: center;
    }
    </style>'''))
    Javascript(filename='base_rate.js')

class BayesQuestion:
    def __init__(self, textbox, previous, submit_callback, back=True):
        self.displayed = False
        self.label = wd.HTMLMath(value='')
        self.textbox = textbox
        self.submit = wd.Button(description='Next')
        self.submit.on_click(self.submitCallback)
        self.back = wd.Button(description='back')
        self.back.on_click(self.backCallback)
        row = [self.textbox, self.submit, self.back] if back else [self.textbox, self.submit]
        self.vbox = wd.VBox([self.label, wd.HBox(row)])
        self.previous = previous
        self.submit_callback = submit_callback

    def display(self, label):
        self.label.value = r'<span style="font-size:16px">' + label + r'</span>'
        if not self.displayed:
            display(self.vbox)
            self.displayed = True
        else:
            self.vbox.layout.display = 'block'

    def disable(self):
        for w in [self.textbox, self.submit, self.back]:
            w.disabled = True

    def enable(self):
        for w in [self.textbox, self.submit, self.back]:
            w.disabled = False

    def hide(self):
        self.vbox.layout.display = 'none'

    def submitCallback(self, sender=None):
        self.disable()
        self.submit_callback()

    def backCallback(self, sender=None):
        self.hide()
        self.previous.enable()

class BayesResult:
    def __init__(self, previous, restart_callback):
        self.displayed = False
        self.label = wd.HTMLMath(value='')
        self.restart = wd.Button(description='Start over')
        self.restart.on_click(self.restartCallback)
        self.back = wd.Button(description='Back')
        self.back.on_click(self.backCallback)
        self.vbox = wd.VBox([self.label, wd.HBox([self.restart, self.back])])
        self.previous = previous
        self.restart_callback = restart_callback

    def display(self, label):
        self.label.value = r'<span style="font-size:16px">' + label + r'</span>'
        if not self.displayed:
            display(self.vbox)
            self.displayed = True
        else:
            self.vbox.layout.display = 'block'

    def hide(self):
        self.vbox.layout.display = 'none'

    def backCallback(self, sender=None):
        self.hide()
        self.previous.enable()

    def restartCallback(self, sender=None):
        self.hide()
        self.restart_callback()

class BayesForm:
    def __init__(self):
        self.q1 = BayesQuestion(wd.Text(), None, self.q1Submit, back=False)
        self.q2 = BayesQuestion(wd.BoundedFloatText(value=0.5, min=0, max=1, step=0.001), self.q1, self.q2Submit)
        self.q3 = BayesQuestion(wd.Text(), self.q2, self.q3Submit)
        self.q4 = BayesQuestion(wd.BoundedFloatText(value=0.5, min=0, max=1, step=0.001), self.q3, self.q4Submit)
        self.q5 = BayesQuestion(wd.BoundedFloatText(value=0.5, min=0, max=1, step=0.001), self.q4, self.q5Submit)
        self.q6 = BayesQuestion(wd.RadioButtons(options=['Positive (B)', 'Negative (¬B)']), self.q5, self.q6Submit)
        self.result = BayesResult(self.q6, self.restart)
        self.q1.display(r'What is the question you are trying to answer? (E.g. Do I have Covid? Is Jack the killer?)<br />We will call this $A$ if yes and $\neg A$ if no.')

    def q1Submit(self):
        self.statement = statementify(self.q1.textbox.value)
        self.q2.display('Without performing any further tests, what is the prior probability that ' + self.statement + '?<br />$P(A)$')

    def q2Submit(self):
        self.q3.display(r'What is the test you are performing? (E.g. PCR test, smoke detector.)<br />We will call this $B$ if positive and $\neg B$ if negative.')

    def q3Submit(self):
        self.q4.display('If ' + self.statement + ', how likely would the ' + self.q3.textbox.value + r' correctly turn up positive?<br />$\text{True positive rate} = P(B|A) = 1-P(\neg B|A) = 1 - \text{False negative rate}$')

    def q4Submit(self):
        text = statementify(self.q1.textbox.value, negate=True)
        self.q5.display('If ' + text + ', how likely would the ' + self.q3.textbox.value + r' correctly turn up negative?<br />$\text{True negative rate} = P(\neg B|\neg A) = 1-P(B|\neg A) = 1 - \text{False positive rate}$')

    def q5Submit(self):
        self.q6.display('What is the actual result of the ' + self.q3.textbox.value + '?')

    def q6Submit(self):
        prior = self.q2.textbox.value
        true_pos_rate = self.q4.textbox.value
        false_pos_rate = 1-self.q5.textbox.value
        pos = self.q6.textbox.value == 'Positive (B)'
        bayes_val = bayes(prior, true_pos_rate, false_pos_rate, pos)
        PBA = '{:.3f}'.format(self.q4.textbox.value)
        PA = '{:.3f}'.format(self.q2.textbox.value)
        PBnA = '(1 - {:.3f})'.format(self.q5.textbox.value)
        PnA = '(1 - {:.3f})'.format(self.q2.textbox.value)
        pos_expr = r'P(A|B) = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|\neg A)P(\neg A)} = \frac{' + PBA + r' \times ' + PA + '}{' + PBA + r' \times ' + PA + ' + ' + PBnA + r' \times ' + PnA + '}'
        PnBA = '(1 - {:.3f})'.format(self.q4.textbox.value)
        PnBnA = '{:.3f}'.format(self.q5.textbox.value)
        neg_expr = r'P(A|\neg B) = \frac{P(\neg B|A)P(A)}{P(\neg B|A)P(A) + P(\neg B|\neg A)P(\neg A)} = \frac{' + PnBA + r' \times ' + PA + '}{' + PnBA + r' \times ' + PA + ' + ' + PnBnA + r' \times ' + PnA + '}'
        expr = pos_expr if pos else neg_expr
        self.result.display('Knowing the test result, the probability that ' + self.statement + ' is<br />$$' + expr + ' = {:.3f}.$$'.format(bayes_val))

    def restart(self, sender=None):
        self.q6.enable()
        for q in [self.q6, self.q5, self.q4, self.q3, self.q2]:
            q.backCallback()