#!/usr/bin/env python
# coding: utf-8

# # Code for the paper- Part 3

# This script handles functional characterization analysis.

# ### Plotting imports and notebook configuration

# In[ ]:


import logging
import os
from hashlib import md5

import nimare

LGR = logging.getLogger(__name__)


# ## Top-level Preparation

# In[ ]:


if os.path.isfile("data/neurosynth_dataset_with_mkda_ma.pkl.gz"):
    LGR.info("Loading existing Dataset.")
    ns_dset = nimare.dataset.Dataset.load(
        "data/neurosynth_dataset_with_mkda_ma.pkl.gz",
    )
    kern = nimare.meta.kernel.MKDAKernel(memory_limit="500mb")
    kern._infer_names(affine=md5(ns_dset.masker.mask_img.affine).hexdigest())
else:
    LGR.info("Generating new Dataset.")
    ns_dset = nimare.dataset.Dataset.load("data/neurosynth_dataset.pkl.gz")
    kern = nimare.meta.kernel.MKDAKernel(memory_limit="500mb")
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

if False:
    LGR.info("Initializing CorrelationDecoder.")
    decoder = nimare.decode.continuous.CorrelationDecoder(
        frequency_threshold=0.001,
        meta_estimator=nimare.meta.MKDAChi2(
            kernel_transformer=kern,
            memory_limit="500mb",
        ),
        target_image="z_desc-specificity",
        features=target_features,
    )
    LGR.info("Fitting CorrelationDecoder.")
    decoder.fit(ns_dset)
    LGR.info("Applying CorrelationDecoder.")
    decoding_results = decoder.transform("results/DerSimonianLaird_est.nii.gz")


# ### Table 3

# In[ ]:


if False:
    decoding_results.sort_values(by="r", ascending=False).to_csv(
        "tables/table_03.tsv",
        sep="\t",
        index_label="feature",
    )


# ### Cleanup

# In[ ]:


if False:
    decoder.save("models/correlation_decoder.pkl.gz")
    del decoder


# ## Listing 16

# ### Preparation

# In[ ]:


amygdala_roi = "data/amygdala_roi.nii.gz"
amygdala_ids = ns_dset.get_studies_by_mask(amygdala_roi)


# ### Listing 16

# In[ ]:


decoder = nimare.decode.discrete.BrainMapDecoder(
    frequency_threshold=0.001,
    u=0.05,
    correction="fdr_bh",
    features=target_features,
)
decoder.fit(ns_dset)
decoding_results = decoder.transform(amygdala_ids)


# ### Table 4

# In[ ]:


decoding_results.sort_values(by="probReverse", ascending=False).to_csv(
    "tables/table_04.tsv",
    sep="\t",
    index_label="feature",
)


# ### Cleanup

# In[ ]:


decoder.save("models/brainmap_decoder.pkl.gz")
del decoder


# ## Listing 17

# In[ ]:


decoder = nimare.decode.discrete.NeurosynthDecoder(
    frequency_threshold=0.001,
    u=0.05,
    correction="fdr_bh",
    features=target_features,
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


decoder.save("models/neurosynth_decoder.pkl.gz")
del decoder


# ## Listing 18

# In[ ]:


decoder = nimare.decode.discrete.ROIAssociationDecoder(
    u=0.05,
    correction="fdr_bh",
    features=target_features,
)
decoder.fit(ns_dset)
decoding_results = decoder.transform(amygdala_roi)


# ### Table 6

# In[ ]:


decoding_results.sort_values(by="r", ascending=False).to_csv(
    "tables/table_06.tsv",
    sep="\t",
    index_label="feature",
)


# ### Cleanup

# In[ ]:


decoder.save("models/roi_association_decoder.pkl.gz")
del decoder
