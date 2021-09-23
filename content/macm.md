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

# Meta-analytic coactivation modeling

+++

```{code-cell} ipython3
:tags: [hide-cell]
# First, import the necessary modules and functions
import os

import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets, image, input_data, plotting

import nimare
from nimare.tests.utils import get_test_data_path

# Define where data files will be located
DATA_DIR = os.path.abspath("../data")

# Now, load the Datasets we will use in this chapter
neurosynth_dset = nimare.dataset.Dataset.load(
    os.path.join(DATA_DIR, "neurosynth_dataset.pkl.gz")
)
```

+++

Meta-analytic coactivation modeling (MACM) {cite:p}`Laird2009-gc,Robinson2010-iv,Eickhoff2010-vx`, also known as meta-analytic connectivity modeling, uses meta-analytic data to measure co-occurrence of activations between brain regions providing evidence of functional connectivity of brain regions across tasks.
In coordinate-based MACM, whole-brain studies within the database are selected based on whether or not they report at least one peak in a region of interest specified for the analysis.
These studies are then subjected to a meta-analysis, often comparing the selected studies to those remaining in the database.
In this way, the significance of each voxel in the analysis corresponds to whether there is greater convergence of foci at the voxel among studies which also report foci in the region of interest than those which do not.

<!-- TODO: Determine appropriate citation style here. -->

MACM results have historically been accorded a similar interpretation to task-related functional connectivity (e.g., {cite:p}`Hok2015-lt,Kellermann2013-en`), although this approach is quite removed from functional connectivity analyses of task fMRI data (e.g., beta-series correlations, psychophysiological interactions, or even seed-to-voxel functional connectivity analyses on task data).
Nevertheless, MACM analyses do show high correspondence with resting-state functional connectivity {cite:p}`Reid2017-ez`.
MACM has been used to characterize the task-based functional coactivation of the cerebellum {cite:p}`Riedel2015-tx`, lateral prefrontal cortex {cite:p}`Reid2016-ba`, fusiform gyrus {cite:p}`Caspers2014-ja`, and several other brain regions.

Within NiMARE, MACMs can be performed by selecting studies in a Dataset based on the presence of activation within a target mask or coordinate-centered sphere.

```{code-cell} ipython3
# Create amygdala mask for MACMs
atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
amyg_val = atlas["labels"].index("Right Amygdala")
amygdala_mask = image.math_img(f"img == {amyg_val}", img=atlas["maps"])
amygdala_mask.to_filename("data/amygdala_roi.nii.gz")

# Create Dataset only containing studies with peaks within the amygdala mask
amygdala_ids = neurosynth_dset.get_studies_by_mask("data/amygdala_roi.nii.gz")
dset_amygdala = neurosynth_dset.slice(amygdala_ids)

# Create Dataset only containing studies with peaks within the sphere ROI
sphere_ids = neurosynth_dset.get_studies_by_coordinate([[24, -2, -20]], r=6)
dset_sphere = neurosynth_dset.slice(sphere_ids)
```

**Listing 10.** Selection of studies from a Dataset based on foci in a mask or sphere.

+++

Once the `Dataset` has been reduced to studies with coordinates within the mask or sphere requested, any of the supported CBMA Estimators can be run.

```{code-cell} ipython3
from nimare import meta
meta_amyg = meta.cbma.ale.ALE(kernel__sample_size=20)
results_amyg = meta_amyg.fit(dset_amygdala)
results_amyg.save_maps(output_dir=DATA_DIR, prefix="ALE_Amygdala")

meta_sphere = meta.cbma.ale.ALE(kernel__sample_size=20)
results_sphere = meta_sphere.fit(dset_sphere)
results_sphere.save_maps(output_dir=DATA_DIR, prefix="ALE_Sphere")
```

**Listing 11.** Example meta-analytic coactivation modeling analyses using the ALE algorithm.

```{code-cell} ipython3
:tags: [hide-input]
meta_results = {
    "Amygdala ALE MACM": op.join(DATA_DIR, "ALE_Amygdala_z.nii.gz"),
    "Sphere ALE MACM": op.join(DATA_DIR, "ALE_Sphere_z.nii.gz"),
}

fig, axes = plt.subplots(figsize=(6, 4), nrows=2)
for i_meta, (name, file_) in enumerate(meta_results.items()):
    display = plotting.plot_stat_map(
        file_,
        annotate=False,
        axes=axes[i_meta],
        cmap="Reds",
        cut_coords=[24, -2, -20],
        draw_cross=False,
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
    "figures/figure_08.svg",
    transparent=True,
    bbox_inches="tight",
    pad_inches=0,
)
fig.savefig(
    "figures/figure_08_lowres.png",
    transparent=True,
    bbox_inches="tight",
    pad_inches=0,
)
fig.show()
```

**Figure 8.** One statistical map plot for each of the MACMs, generated with nilearn's `plot_stat_map`.
