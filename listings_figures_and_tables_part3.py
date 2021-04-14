#!/usr/bin/env python
# coding: utf-8

# # Code for the paper- Part 3

# This script handles functional characterization analysis.

# ### Plotting imports and notebook configuration

# In[ ]:


import os
import logging
from hashlib import md5

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
    kern = nimare.meta.kernel.MKDAKernel(low_memory=True)
    kern._infer_names(affine=md5(ns_dset.masker.mask_img.affine).hexdigest())
else:
    LGR.info("Generating new Dataset.")
    ns_dset = nimare.dataset.Dataset.load("data/neurosynth_dataset.pkl.gz")
    kern = nimare.meta.kernel.MKDAKernel(low_memory=True)
    kern._infer_names(affine=md5(ns_dset.masker.mask_img.affine).hexdigest())
    ns_dset.update_path(os.path.abspath("data/ns_dset_maps/"))
    ns_dset = kern.transform(ns_dset, return_type="dataset")
    ns_dset.save("data/neurosynth_dataset_with_mkda_ma.pkl.gz")

# Collect features for decoding
# We use any features that appear in >5% of studies and <95%.
id_cols = ["id", "study_id", "contrast_id"]
frequency_threshold = 0.001
cols = ns_dset.annotations.columns
cols = [c for c in cols if c not in id_cols]
df = ns_dset.annotations.copy()[cols]
n_studies = df.shape[0]
feature_counts = (df >= frequency_threshold).sum(axis=0)
target_features = feature_counts.between(n_studies * 0.05, n_studies * 0.95)
target_features = target_features[target_features]
target_features = target_features.index.values
print(f"{len(target_features)} features selected.", flush=True)


# ## Listing 15

# In[ ]:


if not os.path.isfile("tables/table_03.tsv"):
    LGR.info("Initializing CorrelationDecoder.")
    decoder = nimare.decode.continuous.CorrelationDecoder(
        frequency_threshold=0.001,
        meta_estimator=nimare.meta.MKDAChi2(
            kernel_transformer=kern,
            low_memory=True,
        ),
        target_image="z_desc-specificity",
        features=target_features,
    )
    LGR.info("Fitting CorrelationDecoder.")
    decoder.fit(ns_dset)
    LGR.info("Applying CorrelationDecoder.")
    decoding_results = decoder.transform("data/pain_map.nii.gz")


# ### Figure 10

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
fig.savefig("figures/figure_10.svg")


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


if not os.path.isfile("tables/table_03.tsv"):
    decoder.save("data/correlation_decoder.pkl.gz")
    del decoder


# ## Listing 16

# In[ ]:


if not os.path.isfile("tables/table_04.tsv"):
    decoder = nimare.decode.continuous.CorrelationDistributionDecoder(
        frequency_threshold=0.001,
        target_image=kern.image_type,
        features=target_features,
    )
    decoder.fit(ns_dset)
    decoding_results = decoder.transform("data/pain_map.nii.gz")


# ### Table 4

# In[ ]:


if not os.path.isfile("tables/table_04.tsv"):
    decoding_results.sort_values(by="mean", ascending=False).to_csv(
        "tables/table_04.tsv",
        sep="\t",
        index_label="feature",
    )


# ### Cleanup

# In[ ]:


if not os.path.isfile("tables/table_04.tsv"):
    decoder.save("data/correlation_distribution_decoder.pkl.gz")
    del decoder


# ## Listing 17

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
        features=target_features,
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


if not os.path.isfile("tables/table_05.tsv"):
    decoder.save("data/brainmap_decoder.pkl.gz")
    del decoder


# ## Listing 19

# In[ ]:


if not os.path.isfile("tables/tables_06.tsv"):
    decoder = nimare.decode.discrete.NeurosynthDecoder(
        frequency_threshold=0.001,
        u=0.05,
        correction="fdr_bh",
        features=target_features,
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


if not os.path.isfile("tables/tables_06.tsv"):
    decoder.save("data/neurosynth_decoder.pkl.gz")
    del decoder
