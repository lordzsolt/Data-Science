import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def show_percent(ax, count):
    patches = ax.patches
    size = len(patches)
    if (size % count != 0):
        raise ValueError('The bars cannot be split into' + string(count) + ' groups. Make sure the count parameter is correct')
    
    chunks = np.array_split(patches, count)
    for bars in zip(*chunks):
        heights = list(map(lambda x: x.get_height(), bars))
        total = sum(heights)
        
        for bar, height in zip(bars, heights):
            percent = height / total
            ax.text(bar.get_x() + bar.get_width() / 2, height + 40,
                   '{0:.2%}'.format(percent), ha='center')
            
def plot_learning_curve(estimator, title, X, y, n_jobs = -1, ylim = None, cv = None, train_sizes = np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    
    return plt