#!/usr/bin/env python
# coding: utf-8

# # Automated Annotation

# In[1]:


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
FIG_DIR = os.path.abspath("../images")

# Now, load the Dataset we will use in this chapter
neurosynth_dset_first_500 = nimare.dataset.Dataset.load(
    os.path.join(DATA_DIR, "neurosynth_dataset_first500.pkl.gz")
)


# As mentioned in the discussion of BrainMap ([](content:resources:brainmap)), manually annotating studies in a meta-analytic database can be a time-consuming and labor-intensive process.
# To facilitate more efficient (albeit lower-quality) annotation, NiMARE supports a number of automated annotation approaches.
# These include n-gram term extraction, Cognitive Atlas term extraction, latent Dirichlet allocation, and generalized correspondence latent Dirichlet allocation.
# 
# NiMARE users may download abstracts from PubMed as long as study identifiers in the `Dataset` correspond to PubMed IDs (as in Neurosynth and NeuroQuery).
# Abstracts are much more easily accessible than full article text, so most annotation methods in NiMARE rely on them.
# 
# Below, we use the function {py:func}`nimare.extract.download_abstracts` to download abstracts for the Neurosynth `Dataset`.
# This will attempt to extract metadata about each study in the `Dataset` from PubMed, and then add the abstract available on Pubmed to the `Dataset`'s `texts` attribute, under a new column names "abstract".

# In[2]:


from nimare import dataset, extract

# In order to run this code on nodes without internet access,
# we need this if statement
dataset_file = os.path.join(DATA_DIR, "neurosynth_dataset_first500_with_abstracts.pkl.gz")
if not os.path.isfile(dataset_file):
    neurosynth_dset_first_500 = extract.download_abstracts(
        neurosynth_dset_first_500,
        email="example@email.com",
    )
    neurosynth_dset_first_500.save(dataset_file)
else:
    neurosynth_dset_first_500 = dataset.Dataset.load(dataset_file)


# ## N-gram term extraction
# 
# **N-gram term extraction** refers to the vectorization of text into contiguous sets of words that can be counted as individual tokens.
# The upper limit on the number of words in these tokens is set by the user.
# 
# NiMARE has the function {py:func}`nimare.annotate.text.generate_counts` to extract n-grams from text.
# This method produces either term counts or term frequency- inverse document frequency (tf-idf) values for each of the studies in a `Dataset`.

# In[3]:


from nimare import annotate

counts_df = annotate.text.generate_counts(
    neurosynth_dset_first_500.texts,
    text_column="abstract",
    tfidf=False,
    min_df=10,
    max_df=0.95,
)


# This term count `DataFrame` will be used later, to train a GCLDA model.

# ## Cognitive Atlas term extraction and hierarchical expansion
# 
# **Cognitive Atlas term extraction** leverages the structured nature of the Cognitive Atlas in order to extract counts for individual terms and their synonyms in the ontology, as well as to apply hierarchical expansion to these counts based on the relationships specified between terms.
# This method produces both basic term counts and expanded term counts based on the weights applied to different relationship types present in the ontology.
# 
# First, we must use {py:func}`nimare.extract.download_cognitive_atlas` to download the current version of the Cognitive Atlas ontology.
# This includes both information about individual terms in the ontology and asserted relationships between those terms.
# 
# NiMARE will automatically attempt to extrapolate likely alternate forms of each term in the ontology, in order to make extraction easier.
# For example,

# In[4]:


cogatlas = extract.download_cognitive_atlas(data_dir=DATA_DIR, overwrite=False)
id_df = pd.read_csv(cogatlas["ids"])
rel_df = pd.read_csv(cogatlas["relationships"])

cogat_counts_df, rep_text_df = annotate.cogat.extract_cogat(
    neurosynth_dset_first_500.texts, id_df, text_column="abstract"
)


# In[5]:


# Define a weighting scheme.
# In this scheme, observed terms will also count toward any
# hypernyms (isKindOf), holonyms (isPartOf), and parent categories (inCategory)
# as well.
weights = {"isKindOf": 1, "isPartOf": 1, "inCategory": 1}
expanded_df = annotate.cogat.expand_counts(cogat_counts_df, rel_df, weights)

# Sort by total count and reduce for better visualization
series = expanded_df.sum(axis=0)
series = series.sort_values(ascending=False)
series = series[series > 0]
columns = series.index.tolist()


# In[6]:


# Raw counts
fig, axes = plt.subplots(figsize=(16, 16), nrows=2, sharex=True)
pos = axes[0].imshow(
    cogat_counts_df[columns].values,
    aspect="auto",
    vmin=0,
    vmax=np.max(expanded_df.values),
)
fig.colorbar(pos, ax=axes[0])
axes[0].set_title("Counts Before Expansion", fontsize=20)
axes[0].set_yticks(range(cogat_counts_df.shape[0]))
axes[0].set_yticklabels(cogat_counts_df.index)
axes[0].set_ylabel("Study", fontsize=16)
axes[0].set_xticks(range(len(columns)))
axes[0].set_xticklabels(columns, rotation=90)
axes[0].set_xlabel("Cognitive Atlas Term", fontsize=16)

# Expanded counts
pos = axes[1].imshow(
    expanded_df[columns].values,
    aspect="auto",
    vmin=0,
    vmax=np.max(expanded_df.values),
)
fig.colorbar(pos, ax=axes[1])
axes[1].set_title("Counts After Expansion", fontsize=20)
axes[1].set_yticks(range(cogat_counts_df.shape[0]))
axes[1].set_yticklabels(cogat_counts_df.index)
axes[1].set_ylabel("Study", fontsize=16)
axes[1].set_xticks(range(len(columns)))
axes[1].set_xticklabels(columns, rotation=90)
axes[1].set_xlabel("Cognitive Atlas Term", fontsize=16)

fig.tight_layout()
fig.show()


# In[7]:


# Here we delete the recent variables for the sake of reducing memory usage
del cogatlas, id_df, rel_df, cogat_counts_df, rep_text_df
del weights, expanded_df, series, columns


# ## Latent Dirichlet allocation
# 
# **Latent Dirichlet allocation** (LDA) {cite:p}`Blei2003-lh` was originally combined with meta-analytic neuroimaging data in {cite:t}`Poldrack2012-it`.
# LDA is a generative topic model which, for a text corpus, builds probability distributions across documents and words.
# In LDA, each document is considered a mixture of topics.
# This works under the assumption that each document was constructed by first randomly selecting a topic based on the document's probability distribution across topics, and then randomly selecting a word from that topic based on the topic's probability distribution across words.
# While this is not a useful generative model for producing documents, LDA is able to discern cohesive topics of related words.
# {cite:t}`Poldrack2012-it` were able to apply LDA to full texts from neuroimaging articles in order to develop cognitive neuroscience-related topics and to run topic-wise meta-analyses.
# This method produces two sets of probability distributions: (1) the probability of a word given topic and (2) the probability of a topic given article.
# 
# NiMARE uses a Python-based interface to the MALLET Java library {cite:p}`mccallum2002mallet` to implement LDA.
# NiMARE will download MALLET automatically, when necessary.
# 
# Here, we train an LDA model ({py:class}`nimare.annotate.lda.LDAModel`) on the first 500 studies of the Neurosynth `Dataset`, with 100 topics in the model.

# In[8]:


from nimare import annotate

lda_model = annotate.lda.LDAModel(
    neurosynth_dset_first_500.texts,
    text_column="abstract",
    n_topics=100,
    n_iters=10000,
)
lda_model.fit()


# The most important products of training the `LDAModel` object are its `p_word_g_topic_df_` and `p_topic_g_doc_df_` attributes.
# The `p_word_g_topic_df_` attribute is a `pandas` `DataFrame` in which each row corresponds to a topic and each column corresponds to a term (n-gram) extracted from the `Dataset`'s texts.
# The cells contain weights indicating the probability of selecting the term given that the topic was already selected.
# The `p_topic_g_doc_df_` attribute is also a `DataFrame`.
# In this one, each row corresponds to a study in the `Dataset` and each column is a topic.
# The cell values indicate the probability of selecting a topic when contructing the given study.
# Practically, this indicates the relative proportion with which the topic describes that study.

# In[ ]:


lda_df = lda_model.p_word_g_topic_df_.T
column_names = {c: f"Topic {c}" for c in lda_df.columns}
lda_df = lda_df.rename(columns=column_names)
temp_df = lda_df.copy()
lda_df = pd.DataFrame(columns=lda_df.columns, index=np.arange(10))
lda_df.index.name = "Term"
for col in lda_df.columns:
    top_ten_terms = temp_df.sort_values(by=col, ascending=False).index.tolist()[:10]
    lda_df.loc[:, col] = top_ten_terms

glue("table_lda", lda_df)


# ```{glue:figure} table_lda
# :figwidth: 300px
# :name: "tbl:table_lda"
# :align: center
# 
# The top ten terms for each of the first ten topics in the trained LDA model.
# ```

# In[ ]:


# Here we delete the recent variables for the sake of reducing memory usage
del lda_model, lda_df, temp_df


# ## Generalized correspondence latent Dirichlet allocation
# 
# **Generalized correspondence latent Dirichlet allocation** (GCLDA) is a recently-developed algorithm that trains topics on both article abstracts and coordinates {cite:p}`Rubin2017-rd`.
# GCLDA assumes that topics within the fMRI literature can also be localized to brain regions, in this case modeled as three-dimensional Gaussian distributions.
# These spatial distributions can also be restricted to pairs of Gaussians that are symmetric across brain hemispheres.
# This method produces two sets of probability distributions: the probability of a word given topic (`GCLDAModel.p_word_g_topic_`) and the probability of a voxel given topic (`GCLDAModel.p_voxel_g_topic_`).
# 
# Here we train a GCLDA model ({py:class}`nimare.annotate.gclda.GCLDAModel`) on the first 500 studies of the Neurosynth Dataset.
# The model will include 100 topics, in which the spatial distribution for each topic will be defined as having two Gaussian distributions that are symmetrically localized across the longitudinal fissure.

# In[ ]:


gclda_model = annotate.gclda.GCLDAModel(
    counts_df,
    neurosynth_dset_first_500.coordinates,
    n_regions=2,
    n_topics=100,
    symmetric=True,
    mask=neurosynth_dset_first_500.masker.mask_img,
)
gclda_model.fit(n_iters=2500, loglikely_freq=500)


# In[ ]:


gclda_df = gclda_model.p_word_g_topic_df_.T
column_names = {c: f"Topic {c}" for c in gclda_df.columns}
gclda_df = gclda_df.rename(columns=column_names)
temp_df = gclda_df.copy()
gclda_df = pd.DataFrame(columns=gclda_df.columns, index=np.arange(10))
gclda_df.index.name = "Term"
for col in temp_df.columns:
    top_ten_terms = temp_df.sort_values(by=col, ascending=False).index.tolist()[:10]
    gclda_df.loc[:, col] = top_ten_terms

glue("table_gclda", gclda_df)


# ```{glue:figure} table_gclda
# :figwidth: 300px
# :name: "tbl:table_gclda"
# :align: center
# 
# The top ten terms for each of the first ten topics in the trained GCLDA model.
# ```

# In[ ]:


fig, axes = plt.subplots(nrows=5, figsize=(6, 10))

topic_img_4d = neurosynth_dset_first_500.masker.inverse_transform(gclda_model.p_voxel_g_topic_.T)
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

glue("figure_gclda_topics", fig, display=False)


# ```{glue:figure} figure_gclda_topics
# :figwidth: 150px
# :name: figure_gclda_topics
# :align: center
# 
# Topic weight maps for the first five topics in the GCLDA model.
# ```

# In[ ]:


# Here we delete the recent variables for the sake of reducing memory usage
del gclda_model, temp_df, gclda_df, counts_df

