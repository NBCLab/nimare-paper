#!/usr/bin/env python
# coding: utf-8

# # Code for the paper- Part 1

# This script handles the meta-analyses.

# ### Plotting imports and notebook configuration

# In[ ]:


import os

import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets, image, input_data, plotting

import nimare

FIG_WIDTH = 10
ROW_HEIGHT = 2  # good row height for width of 10


# ## Listing 1

# In[ ]:


sl_dset1 = nimare.io.convert_sleuth_to_dataset(
    "data/contrast-CannabisMinusControl_space-talairach_sleuth.txt"
)
sl_dset2 = nimare.io.convert_sleuth_to_dataset(
    "data/contrast-ControlMinusCannabis_space-talairach_sleuth.txt"
)


# ## Listing 2

# In[ ]:


if os.path.isfile("data/neurosynth_dataset.pkl.gz"):
    ns_dset = nimare.dataset.Dataset.load("data/neurosynth_dataset.pkl.gz")
elif os.path.isfile("data/database.txt"):
    ns_dset = nimare.io.convert_neurosynth_to_dataset(
        "data/database.txt",
        "data/features.txt",
    )
    ns_dset.save("data/neurosynth_dataset.pkl.gz")
else:
    nimare.extract.fetch_neurosynth("data/", unpack=True)
    ns_dset = nimare.io.convert_neurosynth_to_dataset(
        "data/database.txt",
        "data/features.txt",
    )
    ns_dset.save("data/neurosynth_dataset.pkl.gz")


# ## Listing 3

# In[ ]:


mkda_kernel = nimare.meta.kernel.MKDAKernel(r=10)
mkda_ma_maps = mkda_kernel.transform(sl_dset1, return_type="image")
kda_kernel = nimare.meta.kernel.KDAKernel(r=10)
kda_ma_maps = kda_kernel.transform(sl_dset1, return_type="image")
ale_kernel = nimare.meta.kernel.ALEKernel(sample_size=20)
ale_ma_maps = ale_kernel.transform(sl_dset1, return_type="image")


# ### Figure 3

# In[ ]:


max_value = np.max(kda_ma_maps[0].get_fdata()) + 1

fig, axes = plt.subplots(nrows=3, figsize=(FIG_WIDTH, ROW_HEIGHT * 3))
plotting.plot_stat_map(
    mkda_ma_maps[2],
    annotate=False,
    axes=axes[0],
    cmap="Reds",
    cut_coords=[54, -46, 12],
    draw_cross=False,
    figure=fig,
    vmax=max_value,
)
axes[0].set_title("MKDA Kernel")
plotting.plot_stat_map(
    kda_ma_maps[2],
    annotate=False,
    axes=axes[1],
    cmap="Reds",
    cut_coords=[54, -46, 12],
    draw_cross=False,
    figure=fig,
    vmax=max_value,
)
axes[1].set_title("KDA Kernel")
plotting.plot_stat_map(
    ale_ma_maps[2],
    annotate=False,
    axes=axes[2],
    cmap="Reds",
    cut_coords=[54, -46, 12],
    draw_cross=False,
    figure=fig,
)
axes[2].set_title("ALE Kernel")

fig.savefig("figures/figure_03.svg")


# ## Listing 4

# In[ ]:


mkdad_meta = nimare.meta.cbma.mkda.MKDADensity(null_method="analytic")
mkdad_results = mkdad_meta.fit(sl_dset1)


# ## Listing 5

# In[ ]:


ijk = ns_dset.coordinates[["i", "j", "k"]].values
meta = nimare.meta.cbma.ale.SCALE(
    n_iters=10000,
    ijk=ijk,
    low_memory=True,
    kernel__sample_size=20,
)
scale_results = meta.fit(sl_dset1)


# ## Listing 6

# In[ ]:


meta = nimare.meta.cbma.mkda.MKDAChi2()
mkdac_results = meta.fit(sl_dset1, sl_dset2)


# ### Figure 4

# In[ ]:


# Additional meta-analyses for figures
meta = nimare.meta.cbma.mkda.KDA(null_method="analytic")
kda_results = meta.fit(sl_dset1)

meta = nimare.meta.cbma.ale.ALE(null_method="analytic")
ale_results = meta.fit(sl_dset1)

# Meta-analytic maps across estimators
results = [
    mkdad_results,
    mkdac_results,
    kda_results,
    ale_results,
    scale_results,
]
names = ["MKDADensity", "MKDAChi2", "KDA", "ALE", "SCALE"]
fig, axes = plt.subplots(
    figsize=(FIG_WIDTH, ROW_HEIGHT * len(names)),
    nrows=len(names),
)
for i, r in enumerate(results):
    name = names[i]
    if "z" in r.maps.keys():
        stat_img = r.get_map("z", return_type="image")
        cmap = "Reds"
    else:
        stat_img = r.get_map("z_desc-consistency", return_type="image")
        cmap = "RdBu_r"
    plotting.plot_stat_map(
        stat_img,
        annotate=False,
        axes=axes[i],
        cmap=cmap,
        cut_coords=[0, 0, 0],
        draw_cross=False,
        figure=fig,
    )
    axes[i].set_title(name)

fig.savefig("figures/figure_04.svg")


# ## Listing 7

# In[ ]:


from nimare.tests.utils import get_test_data_path

dset_dir = nimare.extract.download_nidm_pain()
dset_file = os.path.join(get_test_data_path(), "nidm_pain_dset.json")
img_dset = nimare.dataset.Dataset(dset_file)
img_dset.update_path(dset_dir)

# Calculate missing images
img_dset.images = nimare.transforms.transform_images(
    img_dset.images,
    target="z",
    masker=img_dset.masker,
    metadata_df=img_dset.metadata,
)
img_dset.images = nimare.transforms.transform_images(
    img_dset.images,
    target="varcope",
    masker=img_dset.masker,
    metadata_df=img_dset.metadata,
)

meta = nimare.meta.ibma.DerSimonianLaird()
dsl_results = meta.fit(img_dset)


# ### Figure 5

# In[ ]:


# Additional meta-analyses for figures
meta = nimare.meta.ibma.Stouffers(use_sample_size=False)
stouffers_results = meta.fit(img_dset)

meta = nimare.meta.ibma.Stouffers(use_sample_size=True)
weighted_stouffers_results = meta.fit(img_dset)

meta = nimare.meta.ibma.Fishers()
fishers_results = meta.fit(img_dset)

meta = nimare.meta.ibma.PermutedOLS()
ols_results = meta.fit(img_dset)

meta = nimare.meta.ibma.WeightedLeastSquares()
wls_results = meta.fit(img_dset)

meta = nimare.meta.ibma.Hedges()
hedges_results = meta.fit(img_dset)

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

meta = nimare.meta.ibma.VarianceBasedLikelihood(method="reml", mask=masker)
vbl_results = meta.fit(img_dset)

meta = nimare.meta.ibma.SampleSizeBasedLikelihood(method="reml", mask=masker)
ssbl_results = meta.fit(img_dset)

# Plot statistical maps from IBMAs
results = [
    dsl_results,
    stouffers_results,
    weighted_stouffers_results,
    fishers_results,
    ols_results,
    wls_results,
    hedges_results,
    vbl_results,
    ssbl_results,
]
names = [
    "DerSimonian-Laird",
    "Stouffer's",
    "Weighted Stouffer's",
    "Fisher's",
    "Ordinary Least Squares",
    "Weighted Least Squares",
    "Hedges'",
    "Variance-Based Likelihood",
    "Sample Size-Based Likelihood",
]

fig, axes = plt.subplots(
    figsize=(FIG_WIDTH, ROW_HEIGHT * len(results)), nrows=len(results)
)
for i, r in enumerate(results):
    img = r.get_map("z")
    plotting.plot_stat_map(
        img,
        annotate=False,
        axes=axes[i],
        cmap="RdBu_r",
        cut_coords=[5, -15, 10],
        draw_cross=False,
        figure=fig,
    )
    axes[i].set_title(names[i])

fig.savefig("figures/figure_05.svg")


# ### Save map for future use

# In[ ]:


dsl_results.get_map("est").to_filename("data/pain_map.nii.gz")


# ## Listing 8

# In[ ]:


mc_corrector = nimare.correct.FWECorrector(
    method="montecarlo", n_iters=10000, n_cores=4
)
mc_results = mc_corrector.transform(mkdad_meta.results)

fdr_corrector = nimare.correct.FDRCorrector(method="indep")
fdr_results = fdr_corrector.transform(mkdad_meta.results)


# ### Figure 6

# In[ ]:


fig, axes = plt.subplots(figsize=(FIG_WIDTH, ROW_HEIGHT * 2), nrows=2)
plotting.plot_stat_map(
    mc_results.get_map("z_level-cluster_corr-FWE_method-montecarlo"),
    annotate=False,
    axes=axes[0],
    draw_cross=False,
    cmap="Reds",
    cut_coords=[0, 0, 0],
    figure=fig,
)
axes[0].set_title("Cluster-level Monte Carlo")
plotting.plot_stat_map(
    fdr_results.get_map("z_corr-FDR_method-indep"),
    annotate=False,
    axes=axes[1],
    draw_cross=False,
    cmap="Reds",
    cut_coords=[0, 0, 0],
    figure=fig,
)
axes[1].set_title("Independent FDR")

fig.savefig("figures/figure_06.svg")


# ## Listing 9

# In[ ]:


kern = nimare.meta.kernel.ALEKernel()
meta = nimare.meta.cbma.ale.ALESubtraction(
    kernel_transformer=kern,
    n_iters=10000,
)
subtraction_results = meta.fit(sl_dset1, sl_dset2)


# ### Figure 7

# In[ ]:


stat_img = subtraction_results.get_map(
    "z_desc-group1MinusGroup2",
    return_type="image",
)
fig, ax = plt.subplots(figsize=(FIG_WIDTH, ROW_HEIGHT))
plotting.plot_stat_map(
    stat_img,
    annotate=False,
    axes=ax,
    cmap="RdBu_r",
    cut_coords=[0, 0, 0],
    draw_cross=False,
    figure=fig,
)
ax.set_title("ALE Subtraction")

fig.savefig("figures/figure_07.svg")


# ## Listing 10

# In[ ]:


# Create amygdala mask for MACMs
atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
amyg_val = atlas["labels"].index("Right Amygdala")
amygdala_mask = image.math_img(f"img == {amyg_val}", img=atlas["maps"])
amygdala_mask.to_filename("data/amygdala_roi.nii.gz")

amygdala_ids = ns_dset.get_studies_by_mask("data/amygdala_roi.nii.gz")
dset_amygdala = ns_dset.slice(amygdala_ids)

sphere_ids = ns_dset.get_studies_by_coordinate([[24, -2, -20]], r=6)
dset_sphere = ns_dset.slice(sphere_ids)


# ## Listing 11

# In[ ]:


meta_amyg = nimare.meta.cbma.ale.ALE(kernel__sample_size=20)
results_amyg = meta_amyg.fit(dset_amygdala)

meta_sphere = nimare.meta.cbma.ale.ALE(kernel__sample_size=20)
results_sphere = meta_sphere.fit(dset_sphere)


# ### Figure 8

# In[ ]:

fig, axes = plt.subplots(figsize=(FIG_WIDTH, ROW_HEIGHT * 2), nrows=2)
plotting.plot_stat_map(
    results_amyg.get_map("z"),
    annotate=False,
    axes=axes[0],
    cmap="Reds",
    cut_coords=[24, -2, -20],
    draw_cross=False,
    figure=fig,
)
axes[0].set_title("Amygdala ALE MACM")
plotting.plot_stat_map(
    results_sphere.get_map("z"),
    annotate=False,
    axes=axes[1],
    cmap="Reds",
    cut_coords=[24, -2, -20],
    draw_cross=False,
    figure=fig,
)
axes[1].set_title("Sphere ALE MACM")

fig.savefig("figures/figure_08.svg")


# In[ ]:


meta_amyg = nimare.meta.cbma.mkda.MKDADensity(null_method="analytic")
results_amyg = meta_amyg.fit(dset_amygdala)

meta_sphere = nimare.meta.cbma.mkda.MKDADensity(null_method="analytic")
results_sphere = meta_sphere.fit(dset_sphere)

fig, axes = plt.subplots(figsize=(FIG_WIDTH, ROW_HEIGHT * 2), nrows=2)
plotting.plot_stat_map(
    results_amyg.get_map("z"),
    annotate=False,
    axes=axes[0],
    cmap="Reds",
    cut_coords=[24, -2, -20],
    draw_cross=False,
    figure=fig,
)
axes[0].set_title("Amygdala MKDA MACM")
plotting.plot_stat_map(
    results_sphere.get_map("z"),
    annotate=False,
    axes=axes[1],
    cmap="Reds",
    cut_coords=[24, -2, -20],
    draw_cross=False,
    figure=fig,
)
axes[1].set_title("Sphere MKDA MACM")

fig.savefig("figures/figure_08a.svg")


# In[ ]:


meta_amyg = nimare.meta.cbma.mkda.KDA()
results_amyg = meta_amyg.fit(dset_amygdala)

meta_sphere = nimare.meta.cbma.mkda.KDA()
results_sphere = meta_sphere.fit(dset_sphere)

fig, axes = plt.subplots(figsize=(FIG_WIDTH, ROW_HEIGHT * 2), nrows=2)
plotting.plot_stat_map(
    results_amyg.get_map("z"),
    annotate=False,
    axes=axes[0],
    cmap="Reds",
    cut_coords=[24, -2, -20],
    draw_cross=False,
    figure=fig,
)
axes[0].set_title("Amygdala KDA MACM")
plotting.plot_stat_map(
    results_sphere.get_map("z"),
    annotate=False,
    axes=axes[1],
    cmap="Reds",
    cut_coords=[24, -2, -20],
    draw_cross=False,
    figure=fig,
)
axes[1].set_title("Sphere KDA MACM")

fig.savefig("figures/figure_08b.svg")


# ## Listing 12

# In[ ]:


# ### Figure 9

# In[ ]:
