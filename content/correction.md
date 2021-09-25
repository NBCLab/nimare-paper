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

# Multiple comparisons correction

+++

```{code-cell} ipython3
:tags: [hide-cell]
# First, import the necessary modules and functions
import os

import matplotlib.pyplot as plt
from myst_nb import glue
from nilearn import plotting

import nimare

# Define where data files will be located
DATA_DIR = os.path.abspath("../data")
FIG_DIR = os.path.abspath("../figures")

# Now, load the Datasets we will use in this chapter
neurosynth_dset = nimare.dataset.Dataset.load(
    os.path.join(DATA_DIR, "neurosynth_dataset.pkl.gz")
)
```

+++

In NiMARE, multiple comparisons correction is separated from each CBMA and IBMA `Estimator`, so that any number of relevant correction methods can be applied after the `Estimator` has been fit to the `Dataset`.
Some correction options, such as the `montecarlo` option for FWE correction, are designed to work specifically with a given `Estimator` (and are indeed implemented within the `Estimator` class, and only called by the `Corrector`).

```{code-cell} ipython3
from nimare import meta, results

mkdad_results = results.MetaResults.load(
    os.path.join(DATA_DIR, "MKDADensity_results.pkl.gz")
)

mc_corrector = correct.FWECorrector(
    method="montecarlo",
    n_iters=10000,
    n_cores=4,
)
mc_results = mc_corrector.transform(mkdad_results)
mc_results.save_maps(output_dir=DATA_DIR, prefix="MKDADensity_FWE")

fdr_corrector = correct.FDRCorrector(method="indep")
fdr_results = fdr_corrector.transform(mkdad_results)
fdr_results.save_maps(output_dir=DATA_DIR, prefix="MKDADensity_FDR")
```

```{code-cell} ipython3
:tags: [hide-cell]
# Here we delete the recent variables for the sake of reducing memory usage
del mkdad_results, mc_corrector, mc_results, fdr_corrector, fdr_results
```

**Listing 8.** Example usage of multiple comparisons correction applied to results from an MKDA meta-analysis.

+++

```{code-cell} ipython3
:tags: [hide-cell]
meta_results = {
    "Cluster-level Monte Carlo": op.join(
        DATA_DIR,
        "MKDADensity_FWE_z_level-cluster_corr-FWE_method-montecarlo.nii.gz"
    ),
    "Independent FDR": op.join(
        DATA_DIR,
        "MKDADensity_FDR_z_corr-FDR_method-indep.nii.gz",
    ),
}

fig, axes = plt.subplots(figsize=(6, 4), nrows=2)

for i_meta, (name, file_) in enumerate(meta_results.items()):
    display = plotting.plot_stat_map(
        file_,
        annotate=False,
        axes=axes[i_meta],
        draw_cross=False,
        cmap="Reds",
        cut_coords=[0, 0, 0],
        figure=fig,
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
    os.path.join(FIG_DIR, "figure_06.svg"),
    transparent=True,
    bbox_inches="tight",
    pad_inches=0,
)
glue("figure_corr_cbma", fig, display=False)
```

```{glue:figure} figure_corr_cbma
:figwidth: 150px
:name: figure_corr_cbma

An array of plots of the corrected statistical maps produced by the different multiple comparisons correction methods.
```
