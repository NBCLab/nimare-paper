#!/usr/bin/env python
# coding: utf-8

# # Code for the paper- Part 3

# This script handles functional characterization analysis.

# ### Plotting imports and notebook configuration

# In[ ]:


import os
import logging

import matplotlib.pyplot as plt
from nilearn import plotting

import nimare

FIG_WIDTH = 10
ROW_HEIGHT = 2  # good row height for width of 10
LGR = logging.getLogger(__name__)


# ## Top-level Preparation

# In[ ]:


if os.path.isfile("data/neurosynth_dataset_with_mkda_ma.pkl.gz"):
    LGR.info("Loading existing Dataset.")
    ns_dset = nimare.dataset.Dataset.load(
        "data/neurosynth_dataset_with_mkda_ma.pkl.gz",
    )
else:
    LGR.info("Generating new Dataset.")
    ns_dset = nimare.dataset.Dataset.load("data/neurosynth_dataset.pkl.gz")
    ns_dset.update_path(os.path.abspath("data/ns_dset_maps/"))
    kern = nimare.meta.kernel.MKDAKernel(low_memory=True)
    ns_dset = kern.transform(ns_dset, return_type="dataset")
    ns_dset.save("data/neurosynth_dataset_with_mkda_ma.pkl.gz")
ns_dset_first500 = ns_dset.slice(ns_dset.ids[:500])


# ## Listing 16

# In[ ]:


if not os.path.isfile("tables/table_03.tsv"):
    LGR.info("Initializing CorrelationDecoder.")
    decoder = nimare.decode.continuous.CorrelationDecoder(
        frequency_threshold=0.001,
        meta_estimator=nimare.meta.cbma.mkda.MKDAChi2(kernel__low_memory=True),
        target_image="z_desc-specificity",
    )
    LGR.info("Fitting CorrelationDecoder.")
    decoder.fit(ns_dset_first500)
    LGR.info("Applying CorrelationDecoder.")
    decoding_results = decoder.transform("data/pain_map.nii.gz")


# ### Figure 11

# In[ ]:


fig, ax = plt.subplots(figsize=(FIG_WIDTH, ROW_HEIGHT))
plotting.plot_stat_map(
    "data/pain_map.nii.gz",
    annotate=False,
    axes=ax,
    cmap="RdBu_r",
    draw_cross=False,
    figure=fig,
)
fig.savefig("figures/figure_11.svg")


# ### Table 3

# In[ ]:


if not os.path.isfile("tables/table_03.tsv"):
    decoding_results.sort_values(by="r", ascending=False).to_csv(
        "tables/table_03.tsv",
        sep="\t",
        index_label="feature",
    )


# ### Cleanup

# In[ ]:


if not os.path.isfile("data/correlation_decoder.pkl.gz"):
    decoder.save("data/correlation_decoder.pkl.gz")
    del decoder


# ## Listing 17

# ### Preparation

# In[ ]:


ns_dset.update_path(os.path.abspath("data/ns_dset_maps/"))

kern = nimare.meta.kernel.MKDAKernel(r=10, value=1, low_memory=True)
ns_dset = kern.transform(ns_dset, return_type="dataset")


# ### Listing 17

# In[ ]:


if not os.path.isfile("tables/table_04.tsv"):
    decoder = nimare.decode.continuous.CorrelationDistributionDecoder(
        frequency_threshold=0.001,
        target_image=kern.image_type,
    )
    decoder.fit(ns_dset)
    decoding_results = decoder.transform("data/pain_map.nii.gz")


# ### Table 4

# In[ ]:


if not os.path.isfile("tables/table_04.tsv"):
    decoding_results.sort_values(by="r", ascending=False).to_csv(
        "tables/table_04.tsv",
        sep="\t",
        index_label="feature",
    )


# ### Cleanup

# In[ ]:


if not os.path.isfile("data/correlation_distribution_decoder.pkl.gz"):
    decoder.save("data/correlation_distribution_decoder.pkl.gz")
    del decoder


# ## Listing 18

# ### Preparation

# In[ ]:


amygdala_ids = ns_dset.get_studies_by_mask("data/amygdala_roi.nii.gz")


# ### Listing 18

# In[ ]:


if not os.path.isfile("tables/table_05.tsv"):
    decoder = nimare.decode.discrete.BrainMapDecoder(
        frequency_threshold=0.001,
        u=0.05,
        correction="fdr_bh",
    )
    decoder.fit(ns_dset)
    decoding_results = decoder.transform(amygdala_ids)


# ### Table 5

# In[ ]:


if not os.path.isfile("tables/table_05.tsv"):
    decoding_results.sort_values(by="probReverse", ascending=False).to_csv(
        "tables/table_05.tsv",
        sep="\t",
        index_label="feature",
    )


# ### Cleanup

# In[ ]:


if not os.path.isfile("data/brainmap_decoder.pkl.gz"):
    decoder.save("data/brainmap_decoder.pkl.gz")
    del decoder


# ## Listing 19

# In[ ]:


if not os.path.isfile("tables/tables_06.tsv"):
    decoder = nimare.decode.discrete.NeurosynthDecoder(
        frequency_threshold=0.001,
        u=0.05,
        correction="fdr_bh",
    )
    decoder.fit(ns_dset)
    decoding_results = decoder.transform(amygdala_ids)


# ### Table 6

# In[ ]:


if not os.path.isfile("tables/tables_06.tsv"):
    decoding_results.sort_values(by="probReverse", ascending=False).to_csv(
        "tables/table_06.tsv",
        sep="\t",
        index_label="feature",
    )


# ### Cleanup

# In[ ]:


if not os.path.isfile("data/neurosynth_decoder.pkl.gz"):
    decoder.save("data/neurosynth_decoder.pkl.gz")
    del decoder
