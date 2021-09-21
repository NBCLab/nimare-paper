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

In NiMARE, multiple comparisons correction is separated from each CBMA and IBMA `Estimator`, so that any number of relevant correction methods can be applied after the `Estimator` has been fit to the `Dataset`. Some correction options, such as the `montecarlo` option for FWE correction, are designed to work specifically with a given `Estimator` (and are indeed implemented within the `Estimator` class, and only called by the `Corrector`).

```{code-cell} ipython3
from nimare.correct import FWECorrector

mc_corrector = FWECorrector(method='montecarlo', n_iters=10000, n_cores=1)
mc_results = mc_corrector.transform(meta.results)

b_corrector = FWECorrector(method='bonferroni')
b_results = b_corrector.transform(meta.results)
```

**Listing 8.** Example usage of multiple comparisons correction applied to results from an MKDA meta-analysis.

+++

**Figure 6.** An array of plots of the corrected statistical maps produced by the different multiple comparisons correction methods.
