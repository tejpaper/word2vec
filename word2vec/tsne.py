from .constants import PATH2CLUSTERS, RANDOM_SEED
from .embedding import Vector

import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from PIL import Image
from io import BytesIO


def tsne(vector: Vector,
         perplexity: int = 30,
         iterations_num: int = 1000,
         is_random: bool = True,
         resolution: int = 1500,
         font_size: float = 0.5
         ) -> Image.Image:

    random_seed = (RANDOM_SEED, None)[is_random]
    tsne_calc = TSNE(2, perplexity=perplexity, n_iter=iterations_num, random_state=random_seed)
    crds = tsne_calc.fit_transform(vector.embedding.weight)

    _, ax = plt.subplots(dpi=resolution, frameon=False)
    ax.set_facecolor('#2b2b2b')
    plt.axis('off')

    limits = np.vstack((crds.max(0), crds.min(0)))[::-1].T
    plt.xlim(limits[0])
    plt.ylim(limits[1])

    with open(PATH2CLUSTERS) as file:
        clusters, colors = json.load(file).values()

    for word, (x, y) in zip(vector.vocabulary.get_itos(), crds):
        for tag, cluster in clusters.items():
            if word in cluster:
                color = colors[tag]
                break
        else:
            color = ('#e3e3e2', "#defb9b")[word.isdigit()]
        plt.text(x, y, word, color=color, size=font_size)

    buf = BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    plt.close()

    return Image.open(buf)
