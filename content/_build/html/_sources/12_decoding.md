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

# Meta-Analytic Functional Decoding

+++

```{code-cell} ipython3
:tags: [hide-cell]
# First, import the necessary modules and functions
import os

import matplotlib.pyplot as plt
from myst_nb import glue
from nilearn import plotting
from repo2data.repo2data import Repo2Data

# Install the data if running locally, or points to cached data if running on neurolibre
DATA_REQ_FILE = os.path.abspath("../binder/data_requirement.json")
repo2data = Repo2Data(DATA_REQ_FILE)
data_path = repo2data.install()
data_path = os.path.join(data_path[0], "data")
```

```{code-cell} ipython3
:tags: [hide-cell]

from nimare import dataset, meta

neurosynth_dset = dataset.Dataset.load(os.path.join(data_path, "neurosynth_dataset.pkl.gz"))

kern = meta.kernel.MKDAKernel(memory_limit="500mb")
neurosynth_dset_first500 = dataset.Dataset.load(
    os.path.join(data_path, "neurosynth_dataset_first500_with_mkda_ma.pkl.gz"),
)

# Collect features for decoding
# We use any features that appear in >10% of studies and <90%.
id_cols = ["id", "study_id", "contrast_id"]
frequency_threshold = 0.001
cols = neurosynth_dset_first500.annotations.columns
cols = [c for c in cols if c not in id_cols]
df = neurosynth_dset_first500.annotations.copy()[cols]
n_studies = df.shape[0]
feature_counts = (df >= frequency_threshold).sum(axis=0)
target_features = feature_counts.between(n_studies * 0.1, n_studies * 0.9)
target_features = target_features[target_features]
target_features = target_features.index.values
print(f"{len(target_features)} features selected.", flush=True)

continuous_map = os.path.join(data_path, "map_to_decode.nii.gz")
amygdala_roi = os.path.join(data_path, "amygdala_roi.nii.gz")
amygdala_ids = neurosynth_dset_first500.get_studies_by_mask(amygdala_roi)
```

+++

Functional decoding performed with meta-analytic data, refers to methods which attempt to predict mental states from neuroimaging data using a large-scale meta-analytic database {cite:p}`Smith2009-wk`.
Such analyses may also be referred to as "informal reverse inference" {cite:p}`Poldrack2011-zp`, "functional characterization analysis" {cite:p}`Bzdok2013-gc,Cieslik2013-kz,Rottschy2013-cd`, "open-ended decoding" {cite:p}`Rubin2017-rd`, or simply "functional decoding" {cite:p}`Amft2015-kw,Bzdok2013-jv,Nickl-Jockschat2015-rg`.
While the terminology is far from standardized, we will refer to this method as **meta-analytic functional decoding** in order to distinguish it from alternative methods like multivariate decoding and model-based decoding {cite:p}`Poldrack2011-zp`.
Meta-analytic functional decoding is often used in conjunction with MACM, meta-analytic clustering, meta-analytic parcellation, and meta-ICA, in order to characterize resulting brain regions, clusters, or components.
Meta-analytic functional decoding models have also been extended for the purpose of **meta-analytic functional encoding**, wherein text is used to generate statistical images {cite:p}`Dockes2018-ug,Nunes2018-du,Rubin2017-rd`.

Four common approaches are correlation-based decoding, dot-product decoding, weight-sum decoding, and chi-square decoding.
We will first discuss continuous decoding methods (i.e., correlation and dot-product), followed by discrete decoding methods (weight-sum and chi-square).

+++

(content:decoding:continuous)=
## Decoding continuous inputs

When decoding unthresholded statistical maps (such as {numref}`figure_map_to_decode`), the most common approaches are to simply correlate the input map with maps from the database, or to compute the dot product between the two maps.
In Neurosynth, meta-analyses are performed for each label (i.e., term or topic) in the database and then the input image is correlated with the resulting unthresholded statistical map from each meta-analysis.
Performing statistical inference on the resulting correlations is not straightforward, however, as voxels display strong spatial correlations, and the true degrees of freedom are consequently unknown (and likely far smaller than the nominal number of voxels).
In order to interpret the results of this decoding approach, users typically select some arbitrary number of top correlation coefficients ahead of time, and use the associated labels to describe the input map.
However, such results should be interpreted with great caution.

```{code-cell} ipython3
:tags: [hide-cell]
fig, ax = plt.subplots(figsize=(10, 6))
plotting.plot_stat_map(continuous_map, axes=ax, figure=fig, annotate=False, draw_cross=False)
glue("figure_map_to_decode", fig, display=False)
```

```{glue:figure} figure_map_to_decode
:name: figure_map_to_decode
:align: center

The unthresholded statistical map that will be used for continuous decoding.
```

This approach can also be applied to an image-based database like NeuroVault, either by correlating input data with meta-analyzed statistical maps, or by deriving distributions of correlation coefficients by grouping statistical maps in the database according to label.
Using these distributions, it is possible to statistically compare labels in order to assess label significance.
NiMARE includes methods for both correlation-based decoding and correlation distribution-based decoding, although the correlation-based decoding is better established and should be preferred over the correlation distribution-based decoding.
As such, we will only show the {py:class}`~nimare.decode.continuous.CorrelationDecoder` here.

```{important}
{py:class}`~nimare.decode.continuous.CorrelationDecoder` currently runs _very_ slowly.
We strongly recommend running it on a subset of labels within the `Dataset`.
It is also quite memory-intensive.

In this example, we have only run the decoder using features appearing in >10% and <90% of the first 500 studies in the `Dataset`.
Additionally, we have pre-generated the results and will simply show the code that _would_ generate
those results, as the decoder requires too much memory for NeuroLibre's servers.
```

```python
from nimare import decode, meta

corr_decoder = decode.continuous.CorrelationDecoder(
    frequency_threshold=0.001,
    meta_estimator=meta.MKDADensity(kernel_transformer=kern, memory_limit=None),
    target_image="z",
    features=target_features,
    memory_limit="500mb",
)
corr_decoder.fit(neurosynth_dset_first500)
corr_df = corr_decoder.transform(continuous_map)
```

```{code-cell} ipython3
import pandas as pd

corr_df = pd.read_table(
    os.path.join(data_path, "correlation_decoder_results.tsv"),
    index_col="feature",
)
```

```{code-cell} ipython3
:tags: [hide-cell]
corr_df = corr_df.reindex(corr_df["r"].abs().sort_values(ascending=False).index)
corr_df = corr_df.iloc[:10]
glue("table_corr", corr_df)
```

```{glue:figure} table_corr
:name: "tbl:table_corr"
:align: center

The top ten terms, sorted by absolute correlation coefficient, from the correlation decoding method.
```

+++

(content:decoding:discrete)=
## Decoding discrete inputs

Decoding regions of interest (ROIs) requires a different approach than decoding unthresholded statistical maps.
One simple approach, used by GCLDA and implemented in the function {py:func}`~nimare.decode.discrete.gclda_decode_roi`, simply sums the `P(topic|voxel)` distribution across all voxels in the ROI in order to produce a value associated with each topic for the ROI.
These **weight sum** values are arbitrarily scaled and cannot be compared across ROIs.
We will not show this method because of its simplicity and the fact that it can only currently be applied to a GCLDA model.

Before we dig into the other decoding methods are are available, let's take a look at the ROI we want to decode.

```{code-cell} ipython3
:tags: [hide-cell]
fig, ax = plt.subplots(figsize=(10, 6))
plotting.plot_roi(amygdala_roi, axes=ax, figure=fig, annotate=False, draw_cross=False)
glue("figure_roi_to_decode", fig, display=False)
```

```{glue:figure} figure_roi_to_decode
:name: figure_roi_to_decode
:align: center

The amygdala region of interest mask that will be used for discrete decoding.
```

One method which relies on correlations, much like the continuous correlation decoder, is the **ROI association** decoding method ({py:class}`~nimare.decode.discrete.ROIAssociationDecoder`), originally implemented in the Neurosynth Python library.
In this method, each study with coordinates in the dataset is convolved with a kernel transformer to produce a modeled activation map.
The resulting modeled activation maps are then masked with a region of interest (i.e., the target of the decoding), and the values are averaged within the ROI.
These averaged modeled activation values are then correlated with the term weights for all labels in the dataset.
This decoding method produces a single correlation coefficient for each of the dataset's labels.

```{important}
Because the `ROIAssociationDecoder` generates modeled activation maps for all of the experiments in the `Dataset`, we will only fit this decoder to the first 500 studies.
```

```{code-cell} ipython3
from nimare import decode

assoc_decoder = decode.discrete.ROIAssociationDecoder(
    amygdala_roi,
    kernel_transformer=kern,
    u=0.05,
    correction="fdr_bh",
)
assoc_decoder.fit(neurosynth_dset_first500)
assoc_df = assoc_decoder.transform()
```

```{code-cell} ipython3
:tags: [hide-cell]
assoc_df = assoc_df.reindex(assoc_df["r"].abs().sort_values(ascending=False).index)
assoc_df = assoc_df.iloc[:10]
glue("table_assoc", assoc_df)
```

```{glue:figure} table_assoc
:name: "tbl:table_assoc"
:align: center

The top ten terms, sorted by absolute correlation coefficient, from the ROI association decoding method.
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del assoc_decoder, assoc_df
```

A more theoretically driven approach to ROI decoding is to use **chi-square-based** methods.
The two methods which use chi-squared tests are the BrainMap decoding method and an adaptation of Neurosynth's meta-analysis method.

In both chi-square-based methods, studies are first selected from a coordinate-based database according to some criterion.
For example, if decoding a region of interest, users might select studies reporting at least one coordinate within 5 mm of the ROI.
Metadata (such as ontological labels) for this subset of studies are then compared to those of the remaining, unselected portion of the database in a confusion matrix.
For each label in the ontology, studies are divided into four groups: selected and label-positive (SS+L+), selected and label-negative (SS+L-), unselected and label-positive (SS-L+), and unselected and label-negative (SS-L-).
Each method then compares these groups in order to evaluate both consistency and specificity of the relationship between the selection criteria and each label, which are evaluated in terms of both statistical significance and effect size.

+++

### BrainMap method

The BrainMap discrete decoding method, implemented in {py:class}`~nimare.decode.discrete.BrainMapDecoder`, compares the distributions of studies with each label within the sample against those in a larger database while accounting for the number of foci from each study.
Broadly speaking, this method assumes that the selection criterion is associated with one peak per study, which means that it is likely only appropriate for selection criteria based around foci, such as regions of interest.
One common analysis, meta-analytic clustering, involves dividing studies within a database into meta-analytic groupings based on the spatial similarity of their modeled activation maps (i.e., study-wise pseudo-statistical maps produced by convolving coordinates with a kernel).
The resulting sets of studies are often functionally decoded in order to build a functional profile associated with each meta-analytic grouping.
While these groupings are defined as subsets of the database, they are not selected based on the location of an individual peak, and so weighting based on the number of foci would be inappropriate.

This decoding method produces four outputs for each label.
First, the distribution of studies in the sample with the label are compared to the distributions of other labels within the sample.
This consistency analysis produces both a measure of statistical significance (i.e., a p-value) and a measure of effect size (i.e., the likelihood of being selected given the presence of the label).
Next, the studies in the sample are compared to the studies in the rest of the database.
This specificity analysis produces a p-value and an effect size measure of the posterior probability of having the label given selection into the sample.
A detailed algorithm description is presented in [](appendices/brainmap_decoding).

```{code-cell} ipython3
brainmap_decoder = decode.discrete.BrainMapDecoder(
    frequency_threshold=0.001,
    u=0.05,
    correction="fdr_bh",
)
brainmap_decoder.fit(neurosynth_dset)
brainmap_df = brainmap_decoder.transform(amygdala_ids)
```

```{code-cell} ipython3
:tags: [hide-cell]
brainmap_df = brainmap_df.reindex(
    brainmap_df["probReverse"].abs().sort_values(ascending=False).index
)
brainmap_df = brainmap_df.iloc[:10]
glue("table_brainmap", brainmap_df)
```

```{glue:figure} table_brainmap
:name: "tbl:table_brainmap"
:align: center

The top ten terms, sorted by reverse-inference posterior probability, from the BrainMap chi-squared decoding method.
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del brainmap_decoder, brainmap_df
```

+++

### Neurosynth method

The implementation of the MKDA Chi-squared meta-analysis method used by Neurosynth is quite similar to BrainMap's method for decoding, if applied to annotations instead of modeled activation values.
This method, implemented in {py:class}`~nimare.decode.discrete.NeurosynthDecoder`, compares the distributions of studies with each label within the sample against those in a larger database, but, unlike the BrainMap method, does not take foci into account.
For this reason, the Neurosynth method would likely be more appropriate for selection criteria not based on regions of interest (e.g., for characterizing meta-analytic groupings from a meta-analytic clustering analysis).
However, the Neurosynth method requires user-provided information that BrainMap does not.
Namely, in order to estimate probabilities for the consistency and specificity analyses with Bayes' Theorem, the Neurosynth method requires a prior probability of a given label.
Typically, a value of 0.5 is used (i.e., the estimated probability that an individual is undergoing a given mental process described by a label, barring any evidence from neuroimaging data, is predicted to be 50%).
This is, admittedly, a poor prediction, which means that probabilities estimated based on this prior are not likely to be accurate, though they may still serve as useful estimates of effect size for the analysis.

Like the BrainMap method, this method produces four outputs for each label.
For the consistency analysis, this method produces both a p-value and a conditional probability of selection given the presence of the label and the prior probability of having the label.
For the specificity analysis, the Neurosynth method produces both a p-value and a posterior probability of presence of the label given selection and the prior probability of having the label.
A detailed algorithm description is presented in [](appendices/neurosynth_decoding).

```{code-cell} ipython3
neurosynth_decoder = decode.discrete.NeurosynthDecoder(
    frequency_threshold=0.001,
    u=0.05,
    correction="fdr_bh",
)
neurosynth_decoder.fit(neurosynth_dset)
neurosynth_df = neurosynth_decoder.transform(amygdala_ids)
```

```{code-cell} ipython3
:tags: [hide-cell]
neurosynth_df = neurosynth_df.reindex(
    neurosynth_df["probReverse"].abs().sort_values(ascending=False).index
)
neurosynth_df = neurosynth_df.iloc[:10]
glue("table_neurosynth", neurosynth_df)
```

```{glue:figure} table_neurosynth
:name: "tbl:table_neurosynth"
:align: center

The top ten terms, sorted by reverse-inference posterior probability, from the Neurosynth chi-squared decoding method.
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del neurosynth_decoder, neurosynth_df
```

+++

In both methods, the database acts as an estimate of the underlying distribution of labels in the real world, such that the probability of having a peak in an ROI given the presence of the label might be interpreted as the probability of a brain activating a specific brain region given that the individual is experiencing a given mental state.
This is a very poor interpretation, given that any database of neuroimaging results will be skewed more toward the interests of the field than the distribution of mental states or processes experienced by humans, which is why decoding must be interpreted with extreme caution.
It is important not to place too much emphasis on the results of functional decoding analyses, although they are very useful in that they can provide a quantitative estimate behind the kinds of interpretations generally included in discussion sections that are normally only backed by informal literature searches or prior knowledge.

The meta-analytic functional decoding methods in NiMARE provide a very rudimentary approach for open-ended decoding (i.e., decoding across a very large range of mental states) that can be used with resources like NeuroVault.
However, standard classification methods have also been applied to datasets from NeuroVault (e.g., {cite:t}`Varoquaux2018-lo`), although these methods do not fall under NiMARE's scope.
