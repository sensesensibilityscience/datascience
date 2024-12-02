import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython import display
import spacy
import lemminflect

total = 1000
ny = 25
c_pos = '#ffa600' # yellow orange
c_neg = '#003f5c' # dark blue
c_pos_dim = '#ffe4b3'
c_neg_dim = '#b3e7ff'

xs = np.arange(total) // ny
ys = np.arange(total) % ny

p_pos = 0.5
p_false_pos = 0.1
p_false_neg = 0.1

highlight_pos = True
highlight_neg = True

def ppv():
    true_pos = p_pos * (1 - p_false_neg)
    false_pos = (1 - p_pos) * p_false_pos
    return true_pos / (true_pos + false_pos)

def npv():
    true_neg = (1 - p_pos) * (1 - p_false_pos)
    false_neg = p_pos * p_false_neg
    return true_neg / (true_neg + false_neg)

@np.vectorize
def c_map(pos, dim):
    return (c_pos_dim if pos else c_neg_dim) if dim else (c_pos if pos else c_neg)

def truth_and_test(p_pos, p_false_pos, p_false_neg):
    arr = np.zeros((2, total))
    n_pos = int(total * p_pos)
    n_false_pos = int(total * (1 - p_pos) * p_false_pos)
    n_false_neg = int(total * p_pos * p_false_neg)
    arr[:,:n_pos] = 1 # the truth
    arr[1,total-n_false_pos:] = 1 # the test result
    arr[1,:n_false_neg] = 0
    return arr

def plot(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text):
    q1 = 'Does this koala have KOALA-21?' if q1_text.value == '' else q1_text.value
    q2 = 'test' if q2_text.value == '' else q2_text.value
    statement = statementify(q1)
    statement_neg = statementify(q1, negate=True)
    p_pos_label.value = f'The prior probability (base rate) that {statement} before doing any test'
    p_false_pos_label.value = f'The probability that the {q2} is positive even though {statement_neg}'
    p_false_neg_label.value = f'The probability that the {q2} is negative even though {statement}'
    ppv_label.value = f'Given that the {q2} is positive, the probability that {statement} is {ppv()*100:.1f}%'
    npv_label.value = f'Given that the {q2} is negative, the probability that {statement_neg} is {npv()*100:.1f}%'
    truth_and_test_results = truth_and_test(p_pos, p_false_pos, p_false_neg)
    dim = (1 - truth_and_test_results[1]) * (not highlight_pos) + truth_and_test_results[1] * (not highlight_neg)
    c_fill = c_map(truth_and_test_results[0], dim)
    c_edge = c_map(truth_and_test_results[1], dim)
    ax.clear()
    ax.scatter(xs, ys, s=50, c=c_fill, edgecolors=c_edge, linewidth=1.5)
    n_false_neg = int(np.sum(truth_and_test_results[0] * (1 - truth_and_test_results[1])))
    n_true_pos = int(np.sum(truth_and_test_results[0] * truth_and_test_results[1]))
    n_true_neg = int(np.sum((1 - truth_and_test_results[0]) * (1 - truth_and_test_results[1])))
    n_false_pos = int(np.sum((1 - truth_and_test_results[0]) * truth_and_test_results[1]))
    legend_elements = [
        plt.Line2D([0], [0], marker='.', linestyle='', color=c_pos if highlight_neg else c_pos_dim, label=f'{n_false_neg} false negative{"s" if n_false_neg != 1 else ""}', markersize=15, markeredgewidth=1.5, markeredgecolor=c_neg if highlight_neg else c_neg_dim),
        plt.Line2D([0], [0], marker='.', linestyle='', color=c_pos if highlight_pos else c_pos_dim, label=f'{n_true_pos} true positive{"s" if n_true_pos != 1 else ""}', markersize=15, markeredgewidth=1.5, markeredgecolor=c_pos if highlight_pos else c_pos_dim),
        plt.Line2D([0], [0], marker='.', linestyle='', color=c_neg if highlight_neg else c_neg_dim, label=f'{n_true_neg} true negative{"s" if n_true_neg != 1 else ""}', markersize=15, markeredgewidth=1.5, markeredgecolor=c_neg if highlight_neg else c_neg_dim),
        plt.Line2D([0], [0], marker='.', linestyle='', color=c_neg if highlight_pos else c_neg_dim, label=f'{n_false_pos} false positive{"s" if n_false_pos != 1 else ""}', markersize=15, markeredgewidth=1.5, markeredgecolor=c_pos if highlight_pos else c_pos_dim)
    ]
    ax.legend(handles=legend_elements, ncol=4, loc="upper center", borderaxespad=-1.5)
    ax.set_title(f'Total: {total}', y=1.05)

def update_labels(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text):
    def f(change):
        plot(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text)
    return f

def update_p_pos(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text):
    def f(change):
        global p_pos
        p_pos = change['new'] / 100
        plot(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text)
    return f

def update_p_false_pos(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text):
    def f(change):
        global p_false_pos
        p_false_pos = change['new'] / 100
        plot(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text)
    return f

def update_p_false_neg(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text):
    def f(change):
        global p_false_neg
        p_false_neg = change['new'] / 100
        plot(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text)
    return f

def update_highlight_pos(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text):
    def f(change):
        global highlight_pos
        highlight_pos = not highlight_pos
        plot(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text)
    return f

def update_highlight_neg(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text):
    def f(change):
        global highlight_neg
        highlight_neg = not highlight_neg
        plot(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text)
    return f

nlp = spacy.load('en_core_web_md')

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

q1_label = widgets.Label(value='What is the question you are trying to answer?')
q1_text = widgets.Text(placeholder='e.g. Do I have Covid? Is Jack the Killer?')
q2_label = widgets.Label(value='What is the test you are performing?')
q2_text = widgets.Text(placeholder='e.g. PCR test')
run_button = widgets.Button(description='Update demo')
questions_hbox = widgets.HBox([
    widgets.VBox([q1_label, q1_text]),
    widgets.VBox([q2_label, q2_text]),
    run_button
], layout={'display': 'flex', 'align_items': 'flex-end'})
p_pos_slider = widgets.FloatSlider(value=50, min=0, max=100, step=0.1, description='Prior probability %', readout=True, readout_format='.1f', layout={'width': '400px'}, style={'description_width': '150px'})
p_pos_label = widgets.Label(value='')
p_pos_hbox = widgets.HBox([p_pos_slider, p_pos_label])
p_false_pos_slider = widgets.FloatSlider(value=10, min=0, max=100, step=0.1, description='False positive rate %', readout=True, readout_format='.1f', layout={'width': '400px'}, style={'description_width': '150px'})
p_false_pos_label = widgets.Label(value='')
p_false_pos_hbox = widgets.HBox([p_false_pos_slider, p_false_pos_label])
p_false_neg_slider = widgets.FloatSlider(value=10, min=0, max=100, step=0.1, description='False negative rate %', readout=True, readout_format='.1f', layout={'width': '400px'}, style={'description_width': '150px'})
p_false_neg_label = widgets.Label(value='')
p_false_neg_hbox = widgets.HBox([p_false_neg_slider, p_false_neg_label])
ppv_label = widgets.Label(value='')
npv_label = widgets.Label(value='')
highlight_pos_toggle = widgets.ToggleButton(value=True, description='Show positive tests')
highlight_neg_toggle = widgets.ToggleButton(value=True, description='Show negative tests')
ppv_hbox = widgets.HBox([highlight_pos_toggle, ppv_label])
npv_hbox = widgets.HBox([highlight_neg_toggle, npv_label])
output = widgets.Output()

plt.rc('axes.spines', top=False, bottom=False, left=False, right=False)
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.toolbar_visible = False
fig.canvas.header_visible = False
fig.canvas.footer_visible = False
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_aspect('equal', 'box')
plot(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text)
plt.tight_layout()
run_button.on_click(update_labels(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text))
p_pos_slider.observe(update_p_pos(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text), names='value')
p_false_pos_slider.observe(update_p_false_pos(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text), names='value')
p_false_neg_slider.observe(update_p_false_neg(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text), names='value')
highlight_pos_toggle.observe(update_highlight_pos(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text), names='value')
highlight_neg_toggle.observe(update_highlight_neg(output, ax, p_pos_label, p_false_pos_label, p_false_neg_label, ppv_label, npv_label, q1_text, q2_text), names='value')

display.display(questions_hbox, p_pos_hbox, p_false_pos_hbox, p_false_neg_hbox, ppv_hbox, npv_hbox, output)