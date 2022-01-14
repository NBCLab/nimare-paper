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

# External Meta-Analytic Resources

+++

```{code-cell} ipython3
:tags: [hide-cell]
# First, import the necessary modules and functions
import os
from pprint import pprint

from repo2data.repo2data import Repo2Data

# Install the data if running locally, or points to cached data if running on neurolibre
DATA_REQ_FILE = os.path.abspath("../binder/data_requirement.json")
repo2data = Repo2Data(DATA_REQ_FILE)
data_path = repo2data.install()
data_path = os.path.join(data_path[0], "data")

# Set an output directory for any files generated during the book building process
out_dir = os.path.abspath("../outputs/")
os.makedirs(out_dir, exist_ok=True)
```

+++

Large-scale meta-analytic databases have made systematic meta-analyses of the neuroimaging literature possible.
These databases combine results from neuroimaging studies, whether represented as coordinates of peak activations or unthresholded statistical images, with important study metadata, such as information about the samples acquired, stimuli used, analyses performed, and mental constructs putatively manipulated.
The two most popular coordinate-based meta-analytic databases are [BrainMap](http://www.brainmap.org) and [Neurosynth](http://neurosynth.org), while the most popular image-based database is [NeuroVault](https://neurovault.org).

The studies archived in these databases may be either manually or automatically annotated—often with reference to a formal ontology or controlled vocabulary.
Ontologies for cognitive neuroscience define what mental states or processes are postulated to be manipulated or measured in experiments, and may also include details of said experiments (e.g.,the cognitive tasks employed), relationships between concepts (e.g., verbal working memory is a kind of working memory), and various other metadata that can be standardized and represented in a machine-readable form {cite:p}`Poldrack2016-ym,Poldrack2010-jz,Turner2012-ai`.
Some of these ontologies are very well-defined, such as expert-generated taxonomies designed specifically to describe only certain aspects of experiments and the relationships between elements within the taxonomy, while others are more loosely defined, in some cases simply building a vocabulary based on which terms are commonly used in cognitive neuroscience articles.

+++

(content:resources:brainmap)=
## BrainMap

**BrainMap** {cite:p}`Fox2005-rt,Fox2002-nv,Laird2005-al` relies on expert annotators to label individual comparisons within studies according to its internally developed ontology, the BrainMap Taxonomy {cite:p}`Fox2005-rt`.
While this approach is likely to be less noisy than an automated annotation method using article text or imaging results to predict content, it is also subject to a number of limitations.
First, there are simply not enough annotators to keep up with the ever-expanding literature.
Second, any development of the underlying ontology has the potential to leave the database outdated.
For example, if a new label is added to the BrainMap Taxonomy, then each study in the full BrainMap database needs to be evaluated for that label before that label can be properly integrated into the database.
Finally, a manually annotated database like BrainMap will be biased by which subdomains within the literature are annotated.
While outside contributors can add and annotate studies to the database, the main source of annotations has been researchers associated with the BrainMap project.

While BrainMap is a semi-closed resource (i.e., a collaboration agreement is required to access the full database), registered users may search the database using the Sleuth search tool, in order to collect samples for meta-analyses.
Sleuth can export these study collections as text files with coordinates.
NiMARE provides a function to import data from Sleuth text files into the NiMARE Dataset format.

The function {py:func}`~nimare.io.convert_sleuth_to_dataset` can be used to convert text files exported from Sleuth into NiMARE `Dataset`s.
Here, we convert two files from a previous publication by NiMARE contributors {cite:p}`yanes2018` into two separate `Dataset`s.

```{code-cell} ipython3
from nimare import io

sleuth_dset1 = io.convert_sleuth_to_dataset(
    os.path.join(data_path, "contrast-CannabisMinusControl_space-talairach_sleuth.txt")
)
sleuth_dset2 = io.convert_sleuth_to_dataset(
    os.path.join(data_path, "contrast-ControlMinusCannabis_space-talairach_sleuth.txt")
)
print(sleuth_dset1)
print(sleuth_dset2)

# Save the Datasets to files for future use
sleuth_dset1.save(os.path.join(out_dir, "sleuth_dset1.pkl.gz"))
sleuth_dset2.save(os.path.join(out_dir, "sleuth_dset2.pkl.gz"))
```

+++

## Neurosynth

**Neurosynth** {cite:p}`Yarkoni2011-dk` uses a combination of web scraping and text mining to automatically harvest neuroimaging studies from the literature and to annotate them based on term frequency within article abstracts.
As a consequence of its relatively crude automated approach, Neurosynth has its own set of limitations.
First, Neurosynth is unable to delineate individual comparisons within studies, and consequently uses the entire paper as its unit of measurement, unlike BrainMap.
This risks conflating directly contrasted comparisons (e.g., A>B and B>A), as well as comparisons which have no relation to one another.
Second, coordinate extraction and annotation are noisy.
Third, annotations automatically performed by Neurosynth are also subject to error, although the reasons behind this are more nuanced and will be discussed later in this paper.
Given Neurosynth's limitations, we recommend that it be used for casual, exploratory meta-analyses rather than for publication-quality analyses.
Nevertheless, while individual meta-analyses should not be published from Neurosynth, many derivative analyses have been performed and published (e.g., {cite:p}`Chang2013-si,De_la_Vega2016-wg,De_la_Vega2018-jc,Poldrack2012-it`).
As evidence of its utility, Neurosynth has been used to define _a priori_ regions of interest (e.g., {cite:p}`Josipovic2014-hx,Zeidman2012-fj,Wager2013-ab`) or perform meta-analytic functional decoding (e.g., {cite:p}`Chen2018-of,Pantelis2015-bq,Tambini2017-iu`) in many first-order (rather than meta-analytic) fMRI studies.

Here, we show code that would download the Neurosynth database from where it is stored (https://github.com/neurosynth/neurosynth-data) and convert it to a NiMARE `Dataset` using {py:func}`~nimare.extract.fetch_neurosynth`, for the first step, and {py:func}`~nimare.io.convert_neurosynth_to_dataset`, for the second.

```{code-cell} ipython3
from nimare import extract

# Download the desired version of Neurosynth from GitHub.
files = extract.fetch_neurosynth(
    data_dir=data_path,
    version="7",
    source="abstract",
    vocab="terms",
    overwrite=False,
)
pprint(files)
neurosynth_db = files[0]
```

```{note}
Converting the large Neurosynth and NeuroQuery datasets to NiMARE {py:class}`~nimare.dataset.Dataset` objects can be a very memory-intensive process.
For the sake of this book, we show how to perform the conversions below, but actually load and use pre-converted `Dataset`s.
```

```python
# Convert the files to a Dataset.
# This may take a while (~10 minutes)
neurosynth_dset = io.convert_neurosynth_to_dataset(
    coordinates_file=neurosynth_db["coordinates"],
    metadata_file=neurosynth_db["metadata"],
    annotations_files=neurosynth_db["features"],
)
print(neurosynth_dset)

# Save the Dataset for later use.
neurosynth_dset.save(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"))
```

Here, we load a pre-generated version of the Neurosynth `Dataset`.

```{code-cell} ipython3
from nimare import dataset

neurosynth_dset = dataset.Dataset.load(os.path.join(data_path, "neurosynth_dataset.pkl.gz"))
print(neurosynth_dset)
```

```{note}
Many of the methods in NiMARE can be very time-consuming or memory-intensive.
Therefore, for the sake of ensuring that the analyses in this article may be reproduced by as many people as possible, we will use a reduced version of the Neurosynth `Dataset`, only containing the first 500 studies, for those methods which may not run easily on the full database.
```

```{code-cell} ipython3
neurosynth_dset_first_500 = neurosynth_dset.slice(neurosynth_dset.ids[:500])
print(neurosynth_dset)

# Save this Dataset for later use.
neurosynth_dset_first_500.save(os.path.join(out_dir, "neurosynth_dataset_first500.pkl.gz"))
```

+++

In addition to a large corpus of coordinates, Neurosynth provides term frequencies derived from article abstracts that can be used as annotations.

+++

One additional benefit to Neurosynth is that it has made available the coordinates for a large number of studies for which the study abstracts are also readily available.
This has made the Neurosynth database a common resource upon which to build other automated ontologies.
Data-driven ontologies which have been developed using the Neurosynth database include the generalized correspondence latent Dirichlet allocation (GCLDA) {cite:p}`Rubin2017-rd` topic model and Deep Boltzmann machines {cite:p}`Monti2016-aq`.

+++

## NeuroQuery

A related resource is **NeuroQuery** {cite:p}`Dockes2020-uv`.
NeuroQuery is an online service for large-scale predictive meta-analysis.
Unlike Neurosynth, which performs statistical inference and produces statistical maps, NeuroQuery is a supervised learning model and produces a prediction of the brain areas most likely to contain activations.
These maps predict locations where studies investigating a given area (determined by the text prompt) are likely to produce activations, but they cannot be used in the same manner as statistical maps from a standard coordinate-based meta-analysis.
In addition to this predictive meta-analytic tool, NeuroQuery also provides a new database of coordinates, text annotations, and metadata via an automated extraction approach that improves on Neurosynth's original methods.

While NiMARE does not currently include an interface to NeuroQuery's predictive meta-analytic method, there are functions for downloading the NeuroQuery database and converting it to NiMARE format, much like Neurosynth.
The functions for downloading the NeuroQuery database and converting it to a `Dataset` are {py:func}`~nimare.extract.fetch_neuroquery` and {py:func}`~nimare.io.convert_neurosynth_to_dataset`, respectively.
We are able to use the same function for converting the database to a `Dataset` for NeuroQuery as Neurosynth because both databases store their data in the same structure.

```{code-cell} ipython3
# Download the desired version of NeuroQuery from GitHub.
files = extract.fetch_neuroquery(
    data_dir=data_path,
    version="1",
    source="combined",
    vocab="neuroquery6308",
    type="tfidf",
    overwrite=False,
)
pprint(files)
neuroquery_db = files[0]
```

```python
# Convert the files to a Dataset.
# This may take a while (~10 minutes)
neuroquery_dset = io.convert_neurosynth_to_dataset(
    coordinates_file=neuroquery_db["coordinates"],
    metadata_file=neuroquery_db["metadata"],
    annotations_files=neuroquery_db["features"],
)
print(neuroquery_dset)

# Save the Dataset for later use.
neuroquery_dset.save(os.path.join(out_dir, "neuroquery_dataset.pkl.gz"))
```

Here, we load a pre-generated version of the NeuroQuery `Dataset`.

```{code-cell} ipython3
neuroquery_dset = dataset.Dataset.load(os.path.join(data_path, "neuroquery_dataset.pkl.gz"))
print(neuroquery_dset)
```

+++

## NeuroVault

**NeuroVault** {cite:p}`Gorgolewski2015-sd` is a public repository of user-uploaded, whole-brain, unthresholded brain maps.
Users may associate their image collections with publications, and can annotate individual maps with labels from the Cognitive Atlas, which is the ontology of choice for NeuroVault.
NiMARE includes a function, {py:func}`~nimare.io.convert_neurovault_to_dataset`, with which users can search for images in NeuroVault, download those images, and convert them into a `Dataset` object.
