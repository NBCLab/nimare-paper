#!/usr/bin/env python
# coding: utf-8

# # Code for the paper- Part 2

# This script handles the automated annotation tools.

# ### Plotting imports and notebook configuration

# In[ ]:


import os

import nimare
import pandas as pd

# ## Top-level Preparation

# In[ ]:


ns_dset = nimare.dataset.Dataset.load("data/neurosynth_dataset.pkl.gz")


# ## Listing 12

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


# ## Listing 13

# In[ ]:


model = nimare.annotate.lda.LDAModel(
    ns_dset_with_abstracts.texts,
    text_column="abstract",
    n_topics=100,
    n_iters=10000,
)
model.fit()


# ### Save model

# In[ ]:


model.save("models/LDAModel.pkl.gz")


# ### Table 1

# In[ ]:


p_word_g_topic_df = model.p_word_g_topic_df_.iloc[:10]
p_word_g_topic_df.to_csv(
    "tables/table_01.tsv",
    sep="\t",
    index_label="topic",
)


# ## Listing 14

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
model.fit(n_iters=2500, loglikely_freq=100)


# ### Save model

# In[ ]:


model.save("models/GCLDAModel.pkl.gz")


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
