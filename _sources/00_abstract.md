# Abstract

We present NiMARE (Neuroimaging Meta-Analysis Research Environment; RRID:SCR_017398; {cite:t}`salo_taylor_2022_6091632`), a Python library for neuroimaging meta-analyses and meta-analysis-related analyses.
NiMARE is an open source, collaboratively-developed package that implements a range of meta-analytic algorithms, including coordinate- and image-based meta-analyses, automated annotation, functional decoding, and meta-analytic coactivation modeling.
By consolidating meta-analytic methods under a common library and syntax, NiMARE makes it straightforward for users to employ the appropriate approach for a given analysis.
In this paper, we describe NiMARE's architecture and the methods implemented in the library.
Additionally, we provide example code and results for each of the available tools in the library.

```{figure} images/figure_00.svg
---
name: top_level_fig
align: center
---
**A graphical representation of tools and methods implemented in NiMARE.**
This diagram outlines six of the most common use-cases for NiMARE.
**(A)** [](05_cbma) (CBMA) is performed by creating a NiMARE `Dataset` with coordinate information stored in the `Dataset.coordinates` attribute, which is then used in a CBMA `Estimator`. This produces a `MetaResult` object with statistical maps, which can then be used in a `Corrector` object for multiple comparisons correction. Once the `Corrector` has been fitted, it will produce a corrected version of the `MetaResult` object, containing updated statistical maps.
**(B)** [](06_ibma) (IBMA) operates similarly to CBMA, except that IBMA `Estimator`s use statistical maps stored in the `Dataset.images` attribute.
**(C)** [](10_macm) (MACM) uses a region of interest to select coordinate-based studies within a `Dataset`, after which the standard CBMA workflow is performed.
**(D)** [](11_annotation) infers labels from textual (and sometimes other) data associated with the `Dataset`, as stored in the `Dataset.texts` attribute. The annotation functions produce labels which may be integrated into the `Dataset` as the `Dataset.annotations` attribute.
**(E)** [Functional decoding of continuous statistical maps](content:decoding:continuous) operates similarly to discrete decoding, in that the input `Dataset` must have both `coordinates` and `annotations` attributes. The `Dataset`, along with an unthresholded statistical map to decode, is provided to the `Decoder` object, which then outputs measures of similarity or associativeness with each label.
**(F)** [Functional decoding of discrete inputs](content:decoding:discrete) applies a selection criterion to a `Dataset` with both `coordinates` and `annotations` attributes, using a `Decoder` object. The decoding algorithm will output measures of similarity or associativeness with each label in the annotations.
```

```{tableofcontents}
```
