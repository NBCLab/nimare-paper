#!/usr/bin/env python
# coding: utf-8

# # Code for the paper- Part 2

# This script handles the automated annotation tools.

# ### Plotting imports and notebook configuration

# In[ ]:


import os

import matplotlib.pyplot as plt
import pandas as pd
from nilearn import image, plotting

import nimare

FIG_WIDTH = 10
ROW_HEIGHT = 2  # good row height for width of 10


# ## Top-level Preparation

# In[ ]:


ns_dset = nimare.dataset.Dataset.load("data/neurosynth_dataset.pkl.gz")


# ## Listing 13

# In[ ]:


if not os.path.isfile("data/neurosynth_dataset_with_abstracts.pkl.gz"):
    ns_dset_with_abstracts = nimare.extract.download_abstracts(
        ns_dset,
        email="example@email.com",
    )
    ns_dset_with_abstracts.save(
        "data/neurosynth_dataset_with_abstracts.pkl.gz",
    )
else:
    ns_dset_with_abstracts = nimare.dataset.Dataset.load(
        "data/neurosynth_dataset_with_abstracts.pkl.gz",
    )


# ## Listing 14

# In[ ]:


model = nimare.annotate.lda.LDAModel(
    ns_dset_with_abstracts.texts,
    text_column="abstract",
    n_topics=100,
    n_iters=10000,
)
model.fit()


# ### Table 1

# In[ ]:


p_word_g_topic_df = model.p_word_g_topic_df_.iloc[:10]
p_word_g_topic_df.to_csv(
    "tables/table_01.tsv",
    sep="\t",
    index_label="topic",
)


# ## Listing 15

# In[ ]:


ns_dset_first_500 = ns_dset_with_abstracts.slice(
    ns_dset_with_abstracts.ids[:500],
)
counts_df = nimare.annotate.text.generate_counts(
    ns_dset_first_500.texts,
    text_column="abstract",
    tfidf=False,
    min_df=10,
    max_df=0.95,
)
model = nimare.annotate.gclda.GCLDAModel(
    counts_df,
    ns_dset_first_500.coordinates,
    n_regions=2,
    n_topics=100,
    symmetric=True,
    mask=ns_dset.masker.mask_img,
)
model.fit(n_iters=500, loglikely_freq=100)


# ### Table 2

# In[ ]:


p_word_g_topic_df = pd.DataFrame(
    data=model.p_word_g_topic_.T,
    columns=model.vocabulary,
)
p_word_g_topic_df = p_word_g_topic_df.iloc[:10]
p_word_g_topic_df.to_csv(
    "tables/table_02.tsv",
    sep="\t",
    index_label="topic",
)


# ### Figure 10

# In[ ]:


fig, axes = plt.subplots(nrows=5, figsize=(FIG_WIDTH, ROW_HEIGHT * 5))

topic_img_4d = ns_dset_first_500.masker.inverse_transform(
    model.p_voxel_g_topic_.T,
)
for i_topic in range(5):
    topic_img = image.index_img(topic_img_4d, index=i_topic)
    plotting.plot_stat_map(
        topic_img,
        annotate=False,
        axes=axes[i_topic],
        cmap="Reds",
        draw_cross=False,
        figure=fig,
    )
    axes[i_topic].set_title(f"Topic {i_topic + 1}")

fig.savefig("figures/figure_10.svg")
