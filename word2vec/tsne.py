from word2vec.embedding import Word2Vec

import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def tsne(word2vec: Word2Vec,
         perplexity: int = 40,
         iterations_num: int = 1000,
         random_seed: int | None = 42,
         font_size: float = 0.5,
         colors_mapping: dict[str, list[str]] | None = None,  # word : color
         ) -> tuple[plt.Figure, plt.Axes]:

    tsne_calc = TSNE(n_components=2,
                     perplexity=perplexity,
                     n_iter=iterations_num,
                     metric='cosine',
                     random_state=random_seed)
    crds = tsne_calc.fit_transform(word2vec.weight.cpu())

    fig, ax = plt.subplots(frameon=False)
    ax.axis('off')

    limits = np.vstack((crds.max(0), crds.min(0)))[::-1].T
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])

    if colors_mapping is None:
        colors_mapping = dict()

    for word, (x, y) in zip(word2vec.vocabulary.get_itos(), crds):
        color = colors_mapping.get(word, 'white')

        if color == 'white':
            path_effects = [pe.withStroke(linewidth=font_size / 5, foreground='black')]
        else:
            path_effects = list()

        ax.text(x, y, word, color=color, size=font_size, path_effects=path_effects)

    plt.close()
    return fig, ax
