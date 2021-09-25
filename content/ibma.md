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

# IBMA methods

+++

```{code-cell} ipython3
:tags: [hide-cell]
# First, import the necessary modules and functions
import os

import matplotlib.pyplot as plt
import numpy as np
from myst_nb import glue
from nilearn import plotting

import nimare

# Define where data files will be located
DATA_DIR = os.path.abspath("../data")
FIG_DIR = os.path.abspath("../figures")

# Now, load the Datasets we will use in this chapter
sleuth_dset1 = nimare.dataset.Dataset.load(
    os.path.join(DATA_DIR, "sleuth_dset1.pkl.gz")
)
```

+++

Image-based meta-analysis (IBMA) methods perform a meta-analysis directly on brain images (either whole-brain or partial) rather than on extracted peaks.
On paper, IBMA is superior to CBMA in virtually all respects, as the availability of analysis-level parameter and variance estimates at all analyzed voxels allows researchers to use the full complement of standard meta-analysis techniques, instead of having to resort to kernel-based or other methods that require additional spatial assumptions.
In principle, given a set of maps that contains no missing values (i.e., where there are _k_ valid pairs of parameter and variance estimates at each voxel), one can simply conduct a voxel-wise version of any standard meta-analysis or meta-regression method commonly used in other biomedical or social science fields.

In practice, the utility of IBMA methods has historically been quite limited, as unthresholded statistical maps have been unavailable for the vast majority of neuroimaging studies.
However, the introduction and rapid adoption of NeuroVault {cite:p}`Gorgolewski2015-sd`, a database for unthresholded statistical images, has made image-based meta-analysis increasingly viable.
Although coverage of the literature remains limited, and IBMAs of maps drawn from the NeuroVault database are likely to omit at least some (and in some cases most) relevant studies due to limited metadata,  we believe the time is ripe for researchers to start including both CBMAs and IBMAs in published meta-analyses, with the aspirational goal of eventually transitioning exclusively to the latter.
To this end, NiMARE supports a range of different IBMA methods, including a number of estimators of the gold standard mixed-effects meta-regression model, as well as several alternative estimators suitable for use when some of the traditional inputs are unavailable.

In the optimal situation, meta-analysts have access to both contrast (i.e., parameter estimate) maps and their associated standard error maps for a number of studies.
With these data, researchers can fit the traditional random-effects meta-regression model using one of several methods that vary in the way they estimate the between-study variance ($\tau^{2}$).
Currently supported estimators include the **DerSimonian-Laird** method {cite:p}`DerSimonian1986-hu`, the **Hedges** method {cite:p}`Hedges1985-ka`, and **maximum-likelihood** (ML) and **restricted maximum-likelihood** (REML) approaches.
NiMARE can also perform fixed-effects meta-regression via weighted least-squares, though there are few IBMA scenarios where a fixed-effects analysis would be indicated.
It is worth noting that the non-likelihood-based estimators (i.e., DerSimonian-Laird and Hedges) have a closed-form solution, and are implemented in an extremely efficient way in NiMARE (i.e., computation is performed on all voxels in parallel).
However, these estimators also produce more biased estimates under typical conditions (e.g., when sample sizes are very small), implying a tradeoff from the user’s perspective.

Alternatively, when users only have access to contrast maps and associated sample sizes, they can use the supported **Sample Size-Based Likelihood** estimator, which assumes that within-study variance is constant across studies, and uses maximum-likelihood or restricted maximum-likelihood to estimate between-study variance, as described in {cite:t}`Sangnawakij2019-mq`.
When users have access only to contrast maps, they can use the **Permuted OLS** estimator, which uses ordinary least squares and employs a max-type permutation scheme for family-wise error correction {cite:p}`Freedman1983-ld,Anderson2001-uc` that has been validated on neuroimaging data {cite:p}`Winkler2014-wh` and relies on the nilearn library.

Finally, when users only have access to z-score maps, they can use either the **Fisher's** {cite:p}`Fisher1925-zh` or the **Stouffer's** {cite:p}`Riley1949-uz` estimators.
When sample size information is available, users may incorporate that information into the Stouffer's method, via the method described in {cite:t}`Zaykin2011-fs`.

```{code-cell} ipython3
from nimare import dataset, extract, transforms

dset_dir = extract.download_nidm_pain()
dset_file = os.path.join(get_test_data_path(), "nidm_pain_dset.json")
img_dset = dataset.Dataset(dset_file)

# Point the Dataset toward the images we've downloaded
img_dset.update_path(dset_dir)

# Calculate missing images
img_transformer = transforms.ImageTransformer(target=["z", "varcope"])
img_dset = img_transformer.transform(img_dset)

img_dset.save(os.path.join(DATA_DIR, "nidm_dset.pkl.gz"))
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del img_transformer
```

```{code-cell} ipython3
from nimare import meta

dsl_meta = meta.ibma.DerSimonianLaird()
dsl_results = dsl_meta.fit(img_dset)

# Save the results for later use
dsl_results.save_maps(output_dir=DATA_DIR, prefix="DerSimonianLaird")
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del dsl_meta, dsl_results
```

**Listing 7.** Transforming images and image-based meta-analysis.

+++

```{code-cell} ipython3
:tags: [hide-cell]
stouffers_meta = meta.ibma.Stouffers(use_sample_size=False)
stouffers_results = stouffers_meta.fit(img_dset)
stouffers_results.save_maps(output_dir=DATA_DIR, prefix="Stouffers")
del stouffers_meta, stouffers_results

weighted_stouffers_meta = meta.ibma.Stouffers(use_sample_size=True)
weighted_stouffers_results = weighted_stouffers_meta.fit(img_dset)
weighted_stouffers_results.save_maps(output_dir=DATA_DIR, prefix="WeightedStouffers")
del weighted_stouffers_meta, weighted_stouffers_results

fishers_meta = meta.ibma.Fishers()
fishers_results = fishers_meta.fit(img_dset)
fishers_results.save_maps(output_dir=DATA_DIR, prefix="Fishers")
del fishers_meta, fishers_results

ols_meta = meta.ibma.PermutedOLS()
ols_results = ols_meta.fit(img_dset)
ols_results.save_maps(output_dir=DATA_DIR, prefix="OLS")
del ols_meta, ols_results

wls_meta = meta.ibma.WeightedLeastSquares()
wls_results = wls_meta.fit(img_dset)
wls_results.save_maps(output_dir=DATA_DIR, prefix="WLS")
del wls_meta, wls_results

hedges_meta = meta.ibma.Hedges()
hedges_results = hedges_meta.fit(img_dset)
hedges_results.save_maps(output_dir=DATA_DIR, prefix="Hedges")
del hedges_meta, hedges_results

# Use atlas for likelihood-based estimators
atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")

# nilearn's NiftiLabelsMasker cannot handle NaNs at the moment,
# and some of the NIDM-Results packs' beta images have NaNs at the edge of the
# brain.
# So, we will create a reduced version of the atlas for this analysis.
nan_mask = image.math_img(
    "~np.any(np.isnan(img), axis=3)", img=img_dset.images["beta"].tolist()
)
nanmasked_atlas = image.math_img(
    "mask * atlas",
    mask=nan_mask,
    atlas=atlas["maps"],
)
masker = input_data.NiftiLabelsMasker(nanmasked_atlas)
del atlas, nan_mask, nanmasked_atlas

vbl_meta = meta.ibma.VarianceBasedLikelihood(method="reml", mask=masker)
vbl_results = vbl_meta.fit(img_dset)
vbl_results.save_maps(output_dir=DATA_DIR, prefix="VBL")
del vbl_meta, vbl_results

ssbl_meta = nimare.meta.ibma.SampleSizeBasedLikelihood(
    method="reml",
    mask=masker,
)
ssbl_results = ssbl_meta.fit(img_dset)
ssbl_results.save_maps(output_dir=DATA_DIR, prefix="SSBL")
del ssbl_meta, ssbl_results, masker
```

+++

```{code-cell} ipython3
:tags: [hide-cell]
meta_results = {
    "DerSimonian-Laird": op.join(DATA_DIR, "DerSimonianLaird_z.nii.gz"),
    "Stouffer's": op.join(DATA_DIR, "Stouffers_z.nii.gz"),
    "Weighted Stouffer's": op.join(DATA_DIR, "WeightedStouffers_z.nii.gz"),
    "Fisher's": op.join(DATA_DIR, "Fishers_z.nii.gz"),
    "Ordinary Least Squares": op.join(DATA_DIR, "OLS_z.nii.gz"),
    "Weighted Least Squares": op.join(DATA_DIR, "WLS_z.nii.gz"),
    "Hedges'": op.join(DATA_DIR, "Hedges_z.nii.gz"),
    "Variance-Based Likelihood": op.join(DATA_DIR, "VBL_z.nii.gz"),
    "Sample Size-Based Likelihood": op.join(DATA_DIR, "SSBL_z.nii.gz"),
}
order = [
    ["Fisher's", "Stouffer's", "Weighted Stouffer's"],
    ["DerSimonian-Laird", "Hedges'", "Weighted Least Squares"],
    ["Ordinary Least Squares", "Variance-Based Likelihood", "Sample Size-Based Likelihood"],
]

fig, axes = plt.subplots(figsize=(18, 6), nrows=3, ncols=3)

for i_row, row_names in enumerate(order):
    for j_col, name in enumerate(row_names):
        file_ = meta_results[name]
        display = plotting.plot_stat_map(
            file_,
            annotate=False,
            axes=axes[i_row, j_col],
            cmap="RdBu_r",
            cut_coords=[5, -15, 10],
            draw_cross=False,
            figure=fig,
        )
        axes[i_row, j_col].set_title(name)

        colorbar = display._cbar
        colorbar_ticks = colorbar.get_ticks()
        if colorbar_ticks[0] < 0:
            new_ticks = [colorbar_ticks[0], 0, colorbar_ticks[-1]]
        else:
            new_ticks = [colorbar_ticks[0], colorbar_ticks[-1]]
        colorbar.set_ticks(new_ticks, update_ticks=True)

fig.savefig(
    os.path.join(FIG_DIR, "figure_05.svg"),
    transparent=True,
    bbox_inches="tight",
    pad_inches=0,
)
glue("figure_uncorr_ibma", fig, display=False)
```

```{glue:figure} figure_uncorr_ibma
:figwidth: 150px
:name: figure_uncorr_ibma

An array of plots of the statistical maps produced by the image-based meta-analysis methods.
The likelihood-based meta-analyses are run on atlases instead of voxelwise.
```
