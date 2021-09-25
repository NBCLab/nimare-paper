---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Automated Annotation

+++

```{code-cell} ipython3
:tags: [hide-cell]
# First, import the necessary modules and functions
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from myst_nb import glue
from nilearn import image, plotting

import nimare

# Define where data files will be located
DATA_DIR = os.path.abspath("../data")
FIG_DIR = os.path.abspath("../figures")

# Now, load the Datasets we will use in this chapter
neurosynth_dset = nimare.dataset.Dataset.load(
    os.path.join(DATA_DIR, "neurosynth_dataset.pkl.gz")
)
neurosynth_dset_first_500 = neurosynth_dset.slice(neurosynth_dset.ids[:500])
neurosynth_dset_first_500.save(
    os.path.join(DATA_DIR, "neurosynth_dataset_first500.pkl.gz")
)
```

+++

As mentioned in the discussion of BrainMap, manually annotating studies in a meta-analytic database can be a time-consuming and labor-intensive process.
To facilitate more efficient (albeit lower-quality) annotation, NiMARE supports a number of automated annotation approaches.
These include n-gram term extraction, Cognitive Atlas term extraction, latent Dirichlet allocation, and generalized correspondence latent Dirichlet allocation.

NiMARE users may download abstracts from PubMed as long as study identifiers in the `Dataset` correspond to PubMed IDs.
Abstracts are much more easily accessible than full article text, so most annotation methods in NiMARE rely on them.
**Listing 12** illustrates how to do this.

```{code-cell} ipython3
from nimare import extract

neurosynth_dset = extract.download_abstracts(
    neurosynth_dset,
    email="example@email.com",
)
neurosynth_dset.save(
    op.join(DATA_DIR, "neurosynth_dataset_with_abstracts.pkl.gz"),
)
```

**Listing 12.** Example usage of the `download_abstracts()` function to download article abstracts from PubMed.

+++

**N-gram term extraction** refers to the vectorization of text into contiguous sets of words that can be counted as individual tokens.
The upper limit on the number of words in these tokens is set by the user.
This method produces either term counts or term frequency- inverse document frequency (tf-idf) values for each of the articles in a `Dataset`.

+++

**Cognitive Atlas term extraction** leverages the structured nature of the Cognitive Atlas in order to extract counts for individual terms and their synonyms in the ontology, as well as to apply hierarchical expansion to these counts based on the relationships specified between terms.
This method produces both basic term counts and expanded term counts based on the weights applied to different relationship types present in the ontology.

+++

**Latent Dirichlet allocation** (LDA) {cite:p}`Blei2003-lh` was originally combined with meta-analytic neuroimaging data in {cite:t}`Poldrack2012-it`.
LDA is a generative topic model which, for a text corpus, builds probability distributions across documents and words.
In LDA, each document is considered a mixture of topics.
This works under the assumption that each document was constructed by first randomly selecting a topic based on the document's probability distribution across topics, and then randomly selecting a word from that topic based on the topic's probability distribution across words.
While this is not a useful generative model for producing documents, LDA is able to discern cohesive topics of related words.
{cite:t}`Poldrack2012-it` were able to apply LDA to full texts from neuroimaging articles in order to develop cognitive neuroscience-related topics and to run topic-wise meta-analyses.
This method produces two sets of probability distributions: (1) the probability of a word given topic and (2) the probability of a topic given article.

NiMARE uses a Python-based interface to the MALLET Java library {cite:p}`mccallum2002mallet` to implement LDA.

```{code-cell} ipython3
from nimare import annotate

lda_model = annotate.lda.LDAModel(
    neurosynth_dset.texts,
    text_column="abstract",
    n_topics=100,
    n_iters=10000,
)
lda_model.fit()
lda_model.save(op.join(DATA_DIR, "LDAModel.pkl.gz"))
```

**Listing 13.** Example training of an LDA topic model.

```{code-cell} ipython3
:tags: [hide-input]

lda_df = lda_model.p_word_g_topic_df_.T
column_names = {c: f"Topic {c}" for c in df.columns}
lda_df = lda_df.rename(columns=column_names)
temp_df = lda_df.copy()
lda_df = pd.DataFrame(columns=lda_df.columns, index=np.arange(10))
lda_df.index.name = "Term"
for col in lda_df.columns:
    top_ten_terms = temp_df.sort_values(
        by=col,
        ascending=False,
    ).index.tolist()[:10]
    lda_df.loc[:, col] = top_ten_terms

glue("table_lda", lda_df)
```

```{glue:figure} table_lda
:figwidth: 300px
:name: "tbl:table_lda"

The top ten terms for each of the first ten topics in the trained LDA model.
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del lda_model, lda_df, temp_df
```

+++

**Generalized correspondence latent Dirichlet allocation** (GCLDA) is a recently-developed algorithm that trains topics on both article abstracts and coordinates {cite:p}`Rubin2017-rd`.
GCLDA assumes that topics within the fMRI literature can also be localized to brain regions, in this case modeled as three-dimensional Gaussian distributions.
These spatial distributions can also be restricted to pairs of Gaussians that are symmetric across brain hemispheres.
This method produces three sets of probability distributions: (1) the probability of a word given topic, (2) the probability of a topic given article, and (3) the probability of a voxel given topic.

```{code-cell} ipython3
counts_df = annotate.text.generate_counts(
    neurosynth_dset_first_500.texts,
    text_column="abstract",
    tfidf=False,
    min_df=10,
    max_df=0.95,
)
gclda_model = annotate.gclda.GCLDAModel(
    counts_df,
    neurosynth_dset_first_500.coordinates,
    n_regions=2,
    n_topics=100,
    symmetric=True,
    mask=neurosynth_dset_first_500.masker.mask_img,
)
gclda_model.fit(n_iters=2500, loglikely_freq=500)

gclda_model.save(op.join(DATA_DIR, "GCLDAModel.pkl.gz"))
```

**Listing 14.** Example training of a GCLDA topic model.

```{code-cell} ipython3
:tags: [hide-input]

gclda_df = gclda_model.p_word_g_topic_df_.T
column_names = {c: f"Topic {c}" for c in gclda_df.columns}
gclda_df = gclda_df.rename(columns=column_names)
temp_df = gclda_df.copy()
gclda_df = pd.DataFrame(columns=gclda_df.columns, index=np.arange(10))
gclda_df.index.name = "Term"
for col in temp_df.columns:
    top_ten_terms = temp_df.sort_values(
        by=col,
        ascending=False,
    ).index.tolist()[:10]
    gclda_df.loc[:, col] = top_ten_terms

glue("table_gclda", gclda_df)
```

```{glue:figure} table_gclda
:figwidth: 300px
:name: "tbl:table_gclda"

The top ten terms for each of the first ten topics in the trained GCLDA model.
```

```{code-cell} ipython3
:tags: [hide-input]
from nilearn import image

fig, axes = plt.subplots(nrows=5, figsize=(6, 10))

topic_img_4d = dset.masker.inverse_transform(gclda_model.p_voxel_g_topic_.T)
# Plot first five topics
for i_topic in range(5):
    topic_img = image.index_img(topic_img_4d, index=i_topic)
    display = plotting.plot_stat_map(
        topic_img,
        annotate=False,
        cmap="Reds",
        draw_cross=False,
        figure=fig,
        axes=axes[i_topic],
    )
    axes[i_topic].set_title(f"Topic {i_topic + 1}")

    colorbar = display._cbar
    colorbar_ticks = colorbar.get_ticks()
    if colorbar_ticks[0] < 0:
        new_ticks = [colorbar_ticks[0], 0, colorbar_ticks[-1]]
    else:
        new_ticks = [colorbar_ticks[0], colorbar_ticks[-1]]
    colorbar.set_ticks(new_ticks, update_ticks=True)

fig.savefig(
    os.path.join(FIG_DIR, "figure_09.svg"),
    transparent=True,
    bbox_inches="tight",
    pad_inches=0,
)
glue("figure_gclda_topics", fig, display=False)
```

```{glue:figure} figure_gclda_topics
:figwidth: 150px
:name: figure_gclda_topics

Topic weight maps for the first five topics in the GCLDA model.
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del gclda_model, temp_df, gclda_df, counts_df
```
