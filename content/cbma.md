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

# CBMA methods

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

Coordinate-based meta-analysis (CBMA) is currently the most popular method for neuroimaging meta-analysis, given that the majority of fMRI papers currently report their findings as peaks of statistically significant clusters in standard space and do not release unthresholded statistical maps.
These peaks indicate where significant results were found in the brain, and thus do not reflect an effect size estimate for each hypothesis test (i.e., each voxel) as one would expect for a typical meta-analysis.
As such, standard methods for effect size-based meta-analysis cannot be applied.
Over the past two decades, a number of algorithms have been developed to determine whether peaks converge across experiments in order to identify locations of consistent or specific activation associated with a given hypothesis {cite:p}`Samartsidis2017-ej,Muller2018-mt`.

Kernel-based methods evaluate convergence of coordinates across studies by first convolving foci with a spatial kernel to produce study-specific modeled activation maps, then combining those modeled activation maps into a sample-wise map, which is compared to a null distribution to evaluate voxel-wise statistical significance.
Additionally, for each of the following approaches, except for SCALE, voxel- or cluster-level multiple comparisons correction may be performed using Monte Carlo simulations or false discovery rate (FDR) {cite:p}`Laird2005-qh` correction. Basic multiple-comparisons correction methods (e.g., Bonferroni correction) are also supported.
**Listing 3** displays a sample code snippet illustrating how to create modeled activation maps for a Dataset using a range of kernel types.

+++

```{figure} ../figures/figure_02.png
:figwidth: 150px
:name: meta_workflow_fig

A flowchart of the typical workflow for coordinate-based meta-analyses in NiMARE.
```

CBMA kernels are available as {py:class}`nimare.meta.kernel.KernelTransformer`s in the {py:mod}`nimare.meta.kernel` module.
There are three standard kernels that are currently available: `MKDAKernel`, `KDAKernel`, and `ALEKernel`.
Each class may be configured with certain parameters when a new object is initialized.
For example, `MKDAKernel` accepts an `r` parameter, which determines the radius of the spheres that will be created around each peak coordinate.
`ALEKernel` automatically uses the sample size associated with each experiment in the `Dataset` to determine the appropriate full-width-at-half-maximum of its Gaussian distribution, as described in {cite:t}`EICKHOFF20122349`; however, users may provide a constant `sample_size` or `fwhm` parameter when sample size information is not available within the `Dataset` metadata.

```{code-cell} ipython3
from nimare.meta import kernel

mkda_kernel = kernel.MKDAKernel(r=10)
mkda_ma_maps = mkda_kernel.transform(sl_dset1)
kda_kernel = kernel.KDAKernel(r=10)
kda_ma_maps = kda_kernel.transform(sl_dset1)
ale_kernel = kernel.ALEKernel(sample_size=20)
ale_ma_maps = ale_kernel.transform(sl_dset1)
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del mkda_kernel, kda_kernel, ale_kernel
```

**Listing 3.** Example usage of available kernel transformers in NiMARE.

+++

```{code-cell} ipython3
:tags: [hide-cell]
# Generate figure
study_idx = 10  # a study with overlapping kernels
max_value = np.max(kda_ma_maps[study_idx].get_fdata()) + 1

ma_maps = {
    "MKDA Kernel": mkda_ma_maps[study_idx],
    "KDA Kernel": kda_ma_maps[study_idx],
    "ALE Kernel": ale_ma_maps[study_idx],
}

fig, axes = plt.subplots(
    nrows=3,
    figsize=(6, 6),
)

for i_meta, (name, img) in enumerate(ma_maps.items()):
    if "ALE" in name:
        vmax = None
    else:
        vmax = max_value

    display = plotting.plot_stat_map(
        img,
        annotate=False,
        axes=axes[i_meta],
        cmap="Reds",
        cut_coords=[5, 0, 29],
        draw_cross=False,
        figure=fig,
        vmax=vmax,
    )
    axes[i_meta].set_title(name)

    colorbar = display._cbar
    colorbar_ticks = colorbar.get_ticks()
    if colorbar_ticks[0] < 0:
        new_ticks = [colorbar_ticks[0], 0, colorbar_ticks[-1]]
    else:
        new_ticks = [colorbar_ticks[0], colorbar_ticks[-1]]
    colorbar.set_ticks(new_ticks, update_ticks=True)

fig.savefig(
    os.path.join(FIG_DIR, "figure_03.svg"),
    transparent=True,
    bbox_inches="tight",
    pad_inches=0,
)
glue("figure_ma_maps", fig, display=False)
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del mkda_ma_maps, kda_ma_maps, ale_ma_maps
```

```{glue:figure} figure_ma_maps
:figwidth: 300px
:name: "figure_ma_maps"

Modeled activation maps produced by NiMARE's `KernelTransformer` classes.
```

+++

**Multilevel kernel density analysis** (MKDA) {cite:p}`Wager2007-jc` is a kernel-based method that convolves each peak from each study with a binary sphere of a set radius.
These peak-specific binary maps are then combined into study-specific maps by taking the maximum value for each voxel.
Study-specific maps are then averaged across the meta-analytic sample.
This averaging is generally weighted by studies’ sample sizes, although other covariates may be included, such as weights based on the type of inference (random or fixed effects) employed in the study’s analysis.
An arbitrary threshold is generally employed to zero-out voxels with very low values, and then a Monte Carlo procedure is used to assess statistical significance, either at the voxel or cluster level.

In NiMARE, the MKDA meta-analyses can be performed with the `nimare.meta.cbma.mkda.MKDADensity` class.
This class, like most other CBMA classes in NiMARE, accepts a `null_method` parameter, which determines how voxel-wise (uncorrected) statistical significance is calculated.
The `null_method` parameter allows two options: "approximate" or "montecarlo."
The "approximate" option builds a histogram-based null distribution of summary-statistic values, which can then be used to determine the associated p-value for _observed_ summary-statistic values (i.e., the values in the meta-analytic map).
The "montecarlo" option builds a null distribution of summary-statistic values by randomly shuffling the coordinates the `Dataset` many times, and computing the summary-statistic values for each permutation.
In general, the "montecarlo" method is slightly more accurate when there are enough permutations, while the "approximate" method is much faster.

```{warning}
Fitting the CBMA `Estimator` to a `Dataset` will produce p-value, z-statistic, and summary-statistic maps, but these are not corrected for multiple comparisons.

When performing a meta-analysis with the goal of statistical inference, you will want to perform multiple comparisons correction with NiMARE's `Corrector`
classes.
Please see the multiple comparisons correction chapter for more information.
```

```{code-cell} ipython3
from nimare.meta.cbma import mkda

mkdad_meta = mkda.MKDADensity(null_method="approximate")
mkdad_results = mkdad_meta.fit(sleuth_dset1)
print(mkdad_results)

# Save the results for later use
mkdad_results.save_maps(output_dir=DATA_DIR, prefix="MKDADensity")
# Save the results *object* as well
mkdad_results.save(os.path.join(DATA_DIR, "MKDADensity_results.pkl.gz"))
```

**Listing 4.** An example MKDA Density meta-analysis in NiMARE.

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del mkdad_meta, mkdad_results
```

+++

Since this is a kernel-based algorithm, the kernel transformer is an optional input to the meta-analytic estimator, and can be controlled in a more fine-grained manner.

```{code-cell} ipython3
# These two approaches (initializing the kernel ahead of time or
# providing the arguments with the kernel__ prefix) are equivalent.
mkda_kernel = kernel.MKDAKernel(r=2)
mkdad_meta = mkda.MKDADensity(kernel_transformer=mkda_kernel)
mkdad_meta = mkda.MKDADensity(
    kernel_transformer=kernel.MKDAKernel,
    kernel__r=2,
)

# A completely different kernel could even be provided, although this is not
# recommended and should only be used for testing algorithms.
mkdad_meta = mkda.MKDADensity(kernel_transformer=kernel.KDAKernel)
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del mkda_kernel, mkdad_meta, mkdad_results
```

+++

**Kernel density analysis** (KDA) {cite:p}`Wager2003-no,Wager2004-ak` is a precursor algorithm that has been replaced in the field by MKDA.
For the sake of completeness, NiMARE also includes a KDA estimator that implements the older KDA algorithm for comparison purposes.
The interface is virtually identical, but since there are few if any legitimate uses of KDA (which models studies as fixed rather than random effects), we do not discuss the algorithm further here.

```{code-cell} ipython3
kda_meta = mkda.KDA(null_method="approximate")
kda_results = kda_meta.fit(sleuth_dset1)
print(kda_results)

# Save the results for later use
kda_results.save_maps(output_dir=DATA_DIR, prefix="KDA")
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del kda_meta, kda_results
```

+++

**Activation likelihood estimation** (ALE) {cite:p}`Eickhoff2012-hk,Turkeltaub2012-no,Turkeltaub2002-dn` assesses convergence of peaks across studies by first generating a modeled activation map for each study, in which each of the experiment’s peaks is convolved with a 3D Gaussian distribution determined by the experiment’s sample size, and then by combining these modeled activation maps across studies into an ALE map, which is compared to an empirical null distribution to assess voxel-wise statistical significance.

```{code-cell} ipython3
from nimare.meta.cbma import ale

ale_meta = ale.ALE()
ale_results = ale_meta.fit(sleuth_dset1)
print(ale_results)

# Save the results for later use
ale_results.save_maps(output_dir=DATA_DIR, prefix="ALE")
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del ale_meta, ale_results
```

+++

**Specific coactivation likelihood estimation** (SCALE) {cite:p}`Langner2014-ei` is an extension of the ALE algorithm developed for meta-analytic coactivation modeling (MACM) analyses.
Rather than comparing convergence of foci within the sample to a null distribution derived under the assumption of spatial randomness within the brain, SCALE assesses whether the convergence at each voxel is greater than in the general literature.
Each voxel in the brain is assigned a null distribution determined based on the base rate of activation for that voxel across an existing coordinate-based meta-analytic database.
This approach allows for the generation of a statistical map for the sample, but no methods for multiple comparisons correction have yet been developed.
While this method was developed to support analysis of joint activation or “coactivation” patterns, it is generic and can be applied to any CBMA; see Other Meta-analytic Approaches below.

```{code-cell} ipython3
# Here we use the coordinates from Neurosynth as our measure of coordinate
# base-rates, because we do not have access to the full BrainMap database.
# However, one assumption of SCALE is that the Dataset being analyzed comes
# from the same source as the database you use for calculating base-rates.
xyz = ns_dset.coordinates[["x", "y", "z"]].values
scale_meta = ale.SCALE(
   n_iters=2500,
   xyz=xyz,
   memory_limit="500mb",
)
scale_results = scale_meta.fit(sleuth_dset1)
print(scale_results)

# Save the results for later use
scale_results.save_maps(output_dir=DATA_DIR, prefix="SCALE")
```

**Listing 5.** An example SCALE meta-analysis.
In this example, we use a larger database, stored as a NiMARE Dataset, to estimate voxel-wise base rates.

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del xyz, scale_meta, scale_results
```

+++

An alternative to the density-based approaches (i.e., MKDA, KDA, ALE, and SCALE) is the **MKDA Chi-squared** extension {cite:p}`Wager2007-jc`.
Though still a kernel-based method in which foci are convolved with a binary sphere and combined within studies, this approach uses voxel-wise chi-squared tests to assess both consistency (i.e., higher convergence of foci within the meta-analytic sample than expected by chance) and specificity (i.e., higher convergence of foci within the meta-analytic sample than detected in an unrelated dataset) of activation.
Such an analysis also requires access to a reference meta-analytic sample or database of studies.
For example, to perform a chi-squared analysis of working memory studies, the researcher will also need a comprehensive set of studies which did not manipulate working memory—ideally one that is matched with the working memory study set on all relevant attributes _except_ the involvement of working memory.

```{code-cell} ipython3
mkdac_meta = mkda.MKDAChi2()
mkdac_results = mkdac_meta.fit(sleuth_dset1, sleuth_dset2)

# Save the results for later use
mkdac_results.save_maps(output_dir="results/", prefix="MKDAChi2")
```

**Listing 6.** An example MKDA Chi-squared meta-analysis.
In this meta-analysis, two samples are compared using voxel-wise chi-squared tests.

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del mkdac_meta, mkdac_results
```

+++

```{code-cell} ipython3
:tags: [hide-cell]

meta_results = {
    "MKDA Density": op.join(DATA_DIR, "MKDADensity_z.nii.gz"),
    "MKDA Chi-Squared": op.join(
      DATA_DIR,
      "MKDAChi2_z_desc-specificity.nii.gz",
    ),
    "KDA": op.join(DATA_DIR, "KDA_z.nii.gz"),
    "ALE": op.join(DATA_DIR, "ALE_z.nii.gz"),
    "SCALE": op.join(DATA_DIR, "SCALE_z.nii.gz"),
}
order = [
    ["MKDA Density", "ALE"],
    ["MKDA Chi-Squared", "SCALE"],
    ["KDA", None]
]

fig, axes = plt.subplots(
    figsize=(12, 6),
    nrows=3,
    ncols=2,
)

for i_row, row_names in enumerate(order):
    for j_col, name in enumerate(row_names):
        if not name:
            axes[i_row, j_col].axis("off")
            continue

        file_ = meta_results[name]
        if "desc-specificity" in file_:
            cmap = "RdBu_r"
        else:
            cmap = "Reds"

        display = plotting.plot_stat_map(
            file_,
            annotate=False,
            axes=axes[i_row, j_col],
            cmap=cmap,
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
    os.path.join(FIG_DIR, "figure_04.svg"),
    transparent=True,
    bbox_inches="tight",
    pad_inches=0,
)
glue("figure_cbma_uncorr", fig, display=False)
```

```{glue:figure} figure_cbma_uncorr
:figwidth: 300px
:name: "figure_cbma_uncorr"

Thresholded results from MKDA Density, KDA, ALE, and SCALE meta-analyses.
```

+++

A number of other coordinate-based meta-analysis algorithms exist which are not yet implemented in NiMARE.
We describe these algorithms briefly in {doc}`../future_directions.md`.
