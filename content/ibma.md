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

Image-based meta-analysis (IBMA) methods perform a meta-analysis directly on brain images (either whole-brain or partial) rather than on extracted peaks.
On paper, IBMA is superior to CBMA in virtually all respects, as the availability of analysis-level parameter and variance estimates at all analyzed voxels allows researchers to use the full complement of standard meta-analysis techniques, instead of having to resort to kernel-based or other methods that require additional spatial assumptions.
In principle, given a set of maps that contains no missing values (i.e., where there are _k_ valid pairs of parameter and variance estimates at each voxel), one can simply conduct a voxel-wise version of any standard meta-analysis or meta-regression method commonly used in other biomedical or social science fields.

In practice, the utility of IBMA methods has historically been quite limited, as unthresholded statistical maps have been unavailable for the vast majority of neuroimaging studies.
However, the introduction and rapid adoption of NeuroVault \cite{Gorgolewski2015-sd}, a database for unthresholded statistical images, has made image-based meta-analysis increasingly viable.
Although coverage of the literature remains limited, and IBMAs of maps drawn from the NeuroVault database are likely to omit at least some (and in some cases most) relevant studies due to limited metadata,  we believe the time is ripe for researchers to start including both CBMAs and IBMAs in published meta-analyses, with the aspirational goal of eventually transitioning exclusively to the latter.
To this end, NiMARE supports a range of different IBMA methods, including a number of estimators of the gold standard mixed-effects meta-regression model, as well as several alternative estimators suitable for use when some of the traditional inputs are unavailable.

In the optimal situation, meta-analysts have access to both contrast (i.e., parameter estimate) maps and their associated standard error maps for a number of studies.
With these data, researchers can fit the traditional random-effects meta-regression model using one of several methods that vary in the way they estimate the between-study variance ($\tau^{2}$).
Currently supported estimators include the **DerSimonian-Laird** method \cite{DerSimonian1986-hu}, the **Hedges** method \cite{Hedges1985-ka}, and **maximum-likelihood** (ML) and **restricted maximum-likelihood** (REML) approaches.
NiMARE can also perform fixed-effects meta-regression via weighted least-squares, though there are few IBMA scenarios where a fixed-effects analysis would be indicated.
It is worth noting that the non-likelihood-based estimators (i.e., DerSimonian-Laird and Hedges) have a closed-form solution, and are implemented in an extremely efficient way in NiMARE (i.e., computation is performed on all voxels in parallel).
However, these estimators also produce more biased estimates under typical conditions (e.g., when sample sizes are very small), implying a tradeoff from the user’s perspective.

Alternatively, when users only have access to contrast maps and associated sample sizes, they can use the supported **Sample Size-Based Likelihood** estimator, which assumes that within-study variance is constant across studies, and uses maximum-likelihood or restricted maximum-likelihood to estimate between-study variance, as described in \cite{Sangnawakij2019-mq}.
When users have access only to contrast maps, they can use the **Permuted OLS** estimator, which uses ordinary least squares and employs a max-type permutation scheme for family-wise error correction \cite{Freedman1983-ld,Anderson2001-uc} that has been validated on neuroimaging data \cite{Winkler2014-wh} and relies on the nilearn library.

Finally, when users only have access to z-score maps, they can use either the **Fisher's** \cite{Fisher1939-zh} or the **Stouffer's** \cite{Riley1949-uz} estimators.
When sample size information is available, users may incorporate that information into the Stouffer's method, via the method described in \cite{Zaykin2011-fs}.

```{code-cell} ipython3
from nimare.meta import ibma

meta = ibma.DerSimonianLaird()
results = meta.fit(img_dset)
```

**Listing 7.** Example usage of an image-based meta-analytic estimator.

+++

**Figure 5.** An array of plots of the statistical maps produced by the image-based meta-analysis methods.
The likelihood-based meta-analyses will be run on atlases instead of voxelwise, which will impact the code and figures.
