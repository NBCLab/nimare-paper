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

Coordinate-based meta-analysis (CBMA) is currently the most popular method for neuroimaging meta-analysis, given that the majority of fMRI papers currently report their findings as peaks of statistically significant clusters in standard space and do not release unthresholded statistical maps. These peaks indicate where significant results were found in the brain, and thus do not reflect an effect size estimate for each hypothesis test (i.e., each voxel) as one would expect for a typical meta-analysis. As such, standard methods for effect size-based meta-analysis cannot be applied. Over the past two decades, a number of algorithms have been developed to determine whether peaks converge across experiments in order to identify locations of consistent or specific activation associated with a given hypothesis \cite{Samartsidis2017-ej,Muller2018-mt}.

Kernel-based methods evaluate convergence of coordinates across studies by first convolving foci with a spatial kernel to produce study-specific modeled activation maps, then combining those modeled activation maps into a sample-wise map, which is compared to a null distribution to evaluate voxel-wise statistical significance. Additionally, for each of the following approaches, except for SCALE, voxel- or cluster-level multiple comparisons correction may be performed using Monte Carlo simulations or false discovery rate (FDR) \cite{Laird2005-qh} correction. Basic multiple-comparisons correction methods (e.g., Bonferroni correction) are also supported. **Listing 3** displays a sample code snippet illustrating how to create modeled activation maps for a Dataset using a range of kernel types.

+++

**Figure 2.** A flowchart of the typical workflow for coordinate-based meta-analyses in NiMARE.

```{code-cell} ipython3
from nimare.meta import kernel

mkda_kernel = kernel.MKDAKernel(r=10)
mkda_res = mkda_kernel.transform(sl_dset1)
kda_kernel = kernel.KDAKernel(r=10)
kda_res = kda_kernel.transform(sl_dset1)
ale_kernel = kernel.ALEKernel(sample_size=20)
ale_res = ale_kernel.transform(sl_dset1)
```

**Listing 3.** Example usage of available kernel transformers in NiMARE.

+++

**Figure 3.** Modeled activation maps produced by **Listing 3.**

+++

**Multilevel kernel density analysis** (MKDA) \cite{Wager2007-jc} is a kernel-based method that convolves each peak from each study with a binary sphere of a set radius. These peak-specific binary maps are then combined into study-specific maps by taking the maximum value for each voxel. Study-specific maps are then averaged across the meta-analytic sample. This averaging is generally weighted by studies’ sample sizes, although other covariates may be included, such as weights based on the type of inference (random or fixed effects) employed in the study’s analysis. An arbitrary threshold is generally employed to zero-out voxels with very low values, and then a Monte Carlo procedure is used to assess statistical significance, either at the voxel or cluster level.

```{code-cell} ipython3
from nimare.meta.cbma import mkda

meta = mkda.MKDADensity()
results = meta.fit(sl_dset1)
```

**Listing 4.** An example MKDA Density meta-analysis in NiMARE.

+++

Since this is a kernel-based algorithm, the kernel transformer is an optional input to the meta-analytic estimator, and can be controlled in a more fine-grained manner.

+++

**Kernel density analysis** (KDA) \cite{Wager2003-no,Wager2004-ak} is a precursor algorithm that has been replaced in the field by MKDA. For the sake of completeness, NiMARE also includes a KDA estimator that implements the older KDA algorithm for comparison purposes. The interface is virtually identical, but since there are few if any legitimate uses of KDA (which models studies as fixed rather than random effects), we do not discuss it further here.

+++

**Activation likelihood estimation** (ALE) \cite{Eickhoff2012-hk,Turkeltaub2012-no,Turkeltaub2002-dn} assesses convergence of peaks across studies by first generating a modeled activation map for each study, in which each of the experiment’s peaks is convolved with a 3D Gaussian distribution determined by the experiment’s sample size, and then by combining these modeled activation maps across studies into an ALE map, which is compared to an empirical null distribution to assess voxel-wise statistical significance.

+++

**Specific coactivation likelihood estimation** (SCALE) \cite{Langner2014-ei} is an extension of the ALE algorithm developed for meta-analytic coactivation modeling (MACM) analyses. Rather than comparing convergence of foci within the sample to a null distribution derived under the assumption of spatial randomness within the brain, SCALE assesses whether the convergence at each voxel is greater than in the general literature. Each voxel in the brain is assigned a null distribution determined based on the base rate of activation for that voxel across an existing coordinate-based meta-analytic database. This approach allows for the generation of a statistical map for the sample, but no methods for multiple comparisons correction have yet been developed. While this method was developed to support analysis of joint activation or “coactivation” patterns, it is generic and can be applied to any CBMA; see Other Meta-analytic Approaches below.

```{code-cell} ipython3
from nimare.meta.cbma import ale

ijk = ns_dset.coordinates[["i", "j", "k"]].values
meta = nimare.meta.cbma.ale.SCALE(
   n_iters=2500,
   ijk=ijk,
   memory_limit="500mb",
)
scale_results = meta.fit(sl_dset1)
```

**Listing 5.** An example SCALE meta-analysis. In this example, we use a larger database, stored as a NiMARE Dataset, to estimate voxel-wise base rates.

+++

An alternative to the density-based approaches (i.e., MKDA, KDA, ALE, and SCALE) is the **MKDA Chi-squared** extension \cite{Wager2007-jc}. Though still a kernel-based method in which foci are convolved with a binary sphere and combined within studies, this approach uses voxel-wise chi-squared tests to assess both consistency (i.e., higher convergence of foci within the meta-analytic sample than expected by chance) and specificity (i.e., higher convergence of foci within the meta-analytic sample than detected in an unrelated dataset) of activation. Such an analysis also requires access to a reference meta-analytic sample or database of studies. For example, to perform a chi-squared analysis of working memory studies, the researcher will also need a comprehensive set of studies which did not manipulate working memory—ideally one that is matched with the working memory study set on all relevant attributes _except_ the involvement of working memory.

```{code-cell} ipython3
from nimare.meta.cbma import mkda

meta = mkda.MKDAChi2()
res = meta.fit(sl_dset1, sl_dset2)
```

**Listing 6.** An example MKDA Chi-squared meta-analysis. In this meta-analysis, two samples are compared using voxel-wise chi-squared tests.

+++

**Figure 4.** Thresholded results from MKDA Density, KDA, ALE, and SCALE meta-analyses.

+++

A number of other coordinate-based meta-analysis algorithms exist which are not yet implemented in NiMARE. We describe these algorithms briefly in the **Future Directions** section below.
