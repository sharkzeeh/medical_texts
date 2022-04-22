import functools
import os
import matplotlib.pyplot as plt
import artm

cv_vocab_path = 'cv_wabbit_v2.vw'

def check_path(fn=None, *, PATH=cv_vocab_path):
    
    if fn is None:
        return lambda fn: check_path(fn, PATH=PATH)

    @functools.wraps(fn)
    def func(*args, **kwargs):
        if not os.path.exists(PATH) or not os.stat(PATH).st_size:
            fn(*args, **kwargs)
        else:
            print(f'{PATH} already exists!')
    return func



def batching(data_path=None, batch_path='batches', text_format='vowpal_wabbit', batches_format='batches', batch_size=256):
    
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)

    if not os.listdir(path=batch_path):
        batch_vectorizer = artm.BatchVectorizer(data_path=data_path,
                                                data_format=text_format, 
                                                target_folder=batch_path, 
                                                batch_size=batch_size)
    else:
        batch_vectorizer = artm.BatchVectorizer(data_path=batch_path,
                                                data_format=batches_format)
        
    return batch_vectorizer


# TM

def tokens_printer(model):
    '''
    shows top tokens for all T topics
    '''
    tokens = model.score_tracker['top_words'].last_tokens
    for topic_name in model.topic_names:
        try:
            print(f'{topic_name}: {", ".join(tokens[topic_name])}', end='\n\n')
        except KeyError:
            pass
        

def show_matrices_sparsity(model):
    print('Phi sparsity:', + round(model.score_tracker['SparsityPhiScore'].last_value, 3))
    print('Theta sparsity:', + round(model.score_tracker['SparsityThetaScore'].last_value, 3))
    
    

def show_training_curve(model):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(model.score_tracker['PerplexityScore'].value)
    ax.set_xticks(np.arange(0, num_collection_passes+1))
    ax.set_xticklabels(np.arange(0, num_collection_passes+1))
    ax.set_xlim([0, num_collection_passes - 1])
    ax.set_xlabel('Num passes')
    ax.set_ylabel('Perplixity score');
    
    
def plot_collection_topic_distribution(bins):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(bins)), list(bins.values()), align='center', edgecolor='k')
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins.keys())
    ax.set_title('Распределение тем в коллекции')
    ax.set_ylabel('# документов в теме')
    ax.set_xlabel('Тема')