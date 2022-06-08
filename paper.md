---
title: "NiMARE: Neuroimaging Meta-Analysis Research Environment"
tags:
  - meta-analysis
  - fMRI
authors:
  - name: Taylor Salo
    affiliation: 1
    orcid: 0000-0001-9813-3167
  - name: Tal Yarkoni
    affiliation: 2
    orcid: 0000-0002-6558-5113
  - name: Thomas E. Nichols
    affiliation: 3
    orcid: 0000-0002-4516-5103
  - name: Jean-Baptiste Poline
    affiliation: 4
    orcid: 0000-0002-9794-749X
  - name: Murat Bilgel
    affiliation: 5
    orcid: 0000-0001-5042-7422
  - name: Katherine L. Bottenhorn
    affiliation: 6
    orcid: 0000-0002-7796-8795
  - name: Dorota Jarecka
    affiliation: 7
    orcid: 0000-0001-8282-2988
  - name: James D. Kent
    affiliation: 8
    orcid: 0000-0002-4892-2659
  - name: Adam Kimbler
    affiliation: 1
    orcid: 0000-0001-5885-9596
  - name: Dylan M. Nielson
    affiliation: 9
    orcid: 0000-0003-4613-6643
  - name: Kendra M. Oudyk
    affiliation: 10
    orcid: 0000-0003-4087-5402
  - name: Julio A. Peraza
    affiliation: 12
    orcid: 0000-0003-3816-5903
  - name: Alexandre Pérez
    affiliation: 10
    orcid: 0000-0003-0556-0763
  - name: Puck C. Reeders
    affiliation: 1
    orcid: 0000-0002-6401-3017
  - name: Julio A. Yanes
    affiliation: 11
    orcid: 0000-0002-6620-4351
  - name: Angela R. Laird
    affiliation: 12
    orcid: 0000-0003-3379-8744
affiliations:
- name: Department of Psychology, Florida International University
  index: 1
- name: Twitter
  index: 2
- name: Big Data Institute, University of Oxford
  index: 3
- name: Neurology and Neurosurgery, McGill University
  index: 4
- name: National Institute on Aging
  index: 5
- name: University of Southern California
  index: 6
- name: Massachusetts Institute of Technology
  index: 7
- name: University of Texas, Austin
  index: 8
- name: National Institute of Mental Health
  index: 9
- name: Montreal Neurological Institute, McGill University
  index: 10
- name: Auburn University
  index: 11
- name: Department of Physics, Florida International University
  index: 12

date: 05 October 2021
bibliography: paper.bib
---

# Summary

We present NiMARE (Neuroimaging Meta-Analysis Research Environment; RRID:SCR_017398), a Python library for neuroimaging meta-analyses and meta-analysis-related analyses [@salo_taylor_2022_6091632].
NiMARE is an open source, collaboratively-developed package that implements a range of meta-analytic algorithms, including coordinate- and image-based meta-analyses, automated annotation, functional decoding, and meta-analytic coactivation modeling.
By consolidating meta-analytic methods under a common library and syntax, NiMARE makes it straightforward for users to employ the appropriate approach for a given analysis.
In this paper, we describe NiMARE's architecture and the methods implemented in the library.
Additionally, we provide example code and results for each of the available tools in the library.

![A graphical representation of tools and methods implemented in NiMARE.\label{top_level_fig}](https://raw.githubusercontent.com/NBCLab/nimare-paper/42d81ecda7353eea51348af773996c7240f10da7/content/images/figure_00.svg)

# Acknowledgements

We would like to thank Yifan Yu and Jérôme Dockès, who provided feedback on the manuscript.

This work was partially funded by the National Institutes of Health (NIH) NIH-NIBIB P41 EB019936 (ReproNim), NIH-NIMH R01 MH083320 (CANDIShare), and NIH RF1 MH120021 (NIDM), the National Institute Of Mental Health under Award Number R01MH096906 (Neurosynth), as well as the Canada First Research Excellence Fund, awarded to McGill University for the Healthy Brains for Healthy Lives initiative and the Brain Canada Foundation with support from Health Canada.

# References
