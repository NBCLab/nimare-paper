#!/usr/bin/env python
# coding: utf-8

# # Code for the paper- Part 3

# This script handles functional characterization analysis.

# ### Plotting imports and notebook configuration

# In[ ]:


import matplotlib.pyplot as plt
from nilearn import plotting

import nimare

FIG_WIDTH = 10
ROW_HEIGHT = 2  # good row height for width of 10


# ## Top-level Preparation

# In[ ]:


ns_dset = nimare.dataset.Dataset.load("data/neurosynth_dataset.pkl.gz")


# ## Listing 16

# In[ ]:


ns_dset.update_path("data/ns_dset_maps/")
decoder = nimare.decode.continuous.CorrelationDecoder(
    frequency_threshold=0.001,
    meta_estimator=nimare.meta.cbma.mkda.MKDAChi2(kernel__low_memory=True),
    target_image="z_desc-specificity",
)
decoder.fit(ns_dset)
decoding_results = decoder.transform("data/pain_map.nii.gz")


# ### Figure 11

# In[ ]:


fig, ax = plt.subplots(figsize=(FIG_WIDTH, ROW_HEIGHT))
plotting.plot_stat_map("data/pain_map.nii.gz", axes=ax)
fig.savefig("figures/figure_11.svg")


# ### Table 3

# In[ ]:


decoding_results.to_csv(
    "tables/table_03.tsv",
    sep="\t",
    index_label="feature",
)


# ### Cleanup

# In[ ]:


decoder.save("data/correlation_decoder.pkl.gz")
del decoder


# ## Listing 17

# ### Preparation

# In[ ]:


ns_dset.update_path("data/ns_dset_maps/")

kern = nimare.meta.kernel.MKDAKernel(r=10, value=1, low_memory=True)
ns_dset = kern.transform(ns_dset, return_type="dataset")


# ### Listing 17

# In[ ]:


decoder = nimare.decode.continuous.CorrelationDistributionDecoder(
    frequency_threshold=0.001,
    target_image=kern.image_type,
)
decoder.fit(ns_dset)
decoding_results = decoder.transform("data/pain_map.nii.gz")


# ### Table 4

# In[ ]:


decoding_results.to_csv(
    "tables/table_04.tsv",
    sep="\t",
    index_label="feature",
)


# ### Cleanup

# In[ ]:


decoder.save("data/correlation_distribution_decoder.pkl.gz")
del decoder


# ## Listing 18

# ### Preparation

# In[ ]:


amygdala_ids = ns_dset.get_studies_by_mask("data/amygdala_roi.nii.gz")


# ### Listing 18

# In[ ]:


decoder = nimare.decode.discrete.BrainMapDecoder(
    frequency_threshold=0.001,
    u=0.05,
    correction="fdr_bh",
)
decoder.fit(ns_dset)
decoding_results = decoder.transform(amygdala_ids)


# ### Table 5

# In[ ]:


decoding_results.sort_values(by="probReverse", ascending=False).to_csv(
    "tables/table_05.tsv",
    sep="\t",
    index_label="feature",
)


# ### Cleanup

# In[ ]:


decoder.save("brainmap_decoder.pkl.gz")
del decoder


# ## Listing 19

# In[ ]:


decoder = nimare.decode.discrete.NeurosynthDecoder(
    frequency_threshold=0.001,
    u=0.05,
    correction="fdr_bh",
)
decoder.fit(ns_dset)
decoding_results = decoder.transform(amygdala_ids)


# ### Table 6

# In[ ]:


decoding_results.sort_values(by="probReverse", ascending=False).to_csv(
    "tables/table_06.tsv",
    sep="\t",
    index_label="feature",
)


# ### Cleanup

# In[ ]:


decoder.save("neurosynth_decoder.pkl.gz")
del decoder
