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

# Package Organization

At present, the package is organized into 14 distinct modules.
`nimare.dataset` defines the `Dataset` class.
`nimare.meta` includes `Estimators` for coordinate- and image-based meta-analysis methods.
`nimare.results` defines the `MetaResult` class, which stores statistical maps produced by meta-analyses.
`nimare.correct` implements `Corrector` classes for family-wise error (FWE) and false discovery rate (FDR) multiple comparisons correction.
`nimare.annotate` implements a range of automated annotation methods, including latent Dirichlet allocation (LDA) and generalized correspondence latent Dirichlet allocation (GCLDA).
`nimare.decode` implements a number of meta-analytic functional decoding and encoding algorithms.
`nimare.io` provides functions for converting alternative meta-analytic dataset structure, such as Sleuth text files or Neurosynth datasets, to NiMARE format.
`nimare.transforms` implements a range of spatial and data type transformations, including a function to generate new images in the `Dataset` from existing image types.
`nimare.extract` provides methods for fetching datasets and models across the internet.
`nimare.generate` includes functions for generating data for internal testing and validation.
`nimare.base` defines a number of base classes used throughout the rest of the package.
Finally, `nimare.stats` and `nimare.utils` are modules for statistical and generic utility functions, respectively.
These modules are summarized in {numref}`table_modules`.

+++

```{table} Summaries of modules in NiMARE.
:name: table_modules

| Module       | Description                                                                                                                                                                                                                                                                                                                                                                                               |
|:-------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dataset`    | This module stores the `Dataset` class, which contains NiMARE datasets.                                                                                                                                                                                                                                                                                                                                   |
| `meta`       | This module contains `Estimators` for image- and coordinate-based meta-analysis algorithms, as well as `KernelTransformers`, which are used in conjunction with coordinate-based methods.                                                                                                                                                                                                                 |
| `results`    | This module stores the `MetaResult` class, which in turn is used to manage statistical maps produced by meta-analytic algorithms.                                                                                                                                                                                                                                                                         |
| `correct`    | This module contains classes for multiple comparisons correction, including `FWECorrector` (family-wise error rate correction) and `FDRCorrector` (false discovery rate correction).                                                                                                                                                                                                                      |
| `annotate`   | This module includes a range of tools for automated annotation of studies. Methods in this module include: topic models, such as latent Dirichlet allocation and generalized correspondence latent Dirichlet allocation; ontology-based annotation, such as Cognitive Atlas term extract from text; and general text-based feature extraction, such as count or tf-idf extraction from text.              |
| `decode`     | This module includes a number of methods for functional characterization analysis, also known as functional decoding. Methods in this module are divided into three groups: discrete, for decoding regions of interest or subsets of the Dataset; continuous, for decoding unthresholded statistical maps; and encoding, for simulating statistical maps from labels.                                     |
| `io`         | This module contains functions for converting common file types, such as Neurosynth- or Sleuth-format files, into NiMARE-compatible formats, such as `Dataset` objects.                                                                                                                                                                                                                                   |
| `transforms` | This module contains classes and functions for converting between common data types. Two important classes in this module are the `ImageTransformer`, which uses available images and metadata to produce new images in a Dataset, and the `ImagesToCoordinates`, which extracts peak coordinates from images in the Dataset, so that image-based studies can be used for coordinate-based meta-analyses. |
| `extract`    | This module contains functions for downloading external resources, such as the Neurosynth dataset and the Cognitive Atlas ontology.                                                                                                                                                                                                                                                                       |
| `stats`      | This module contains miscellaneous statistical methods used throughout the rest of the library.                                                                                                                                                                                                                                                                                                           |
| `generate`   | This module contains functions for generating useful data for internal testing and validation.                                                                                                                                                                                                                                                                                                            |
| `utils`      | This module contains miscellaneous utility functions used throughout the rest of the library.                                                                                                                                                                                                                                                                                                             |
| `workflows`  | This module contains a number of common workflows that can be run from the command line, such as an ALE meta-analysis or a contrast-permutation image-based meta-analysis. All of the workflow functions additionally generate boilerplate text that can be included in manuscript methods sections.                                                                                                      |
| `base`       | This module defines a number of base classes used throughout the rest of the library.                                                                                                                                                                                                                                                                                                                     |
```