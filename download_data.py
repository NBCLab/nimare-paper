"""
Download all data for analyses.
"""
import os.path as op

import neurosynth

import nimare
from nimare import annotate

import utils

DATA_DIR = 'raw_data/'

# The Neurosynth database
neurosynth.dataset.download(path=DATA_DIR, unpack=True)
dset = nimare.io.convert_neurosynth_to_dset(
    op.join(DATA_DIR, 'database.txt'),
    op.join(DATA_DIR, 'features.txt'))
dset = annotate.text.download_abstracts(dset, email='tsalo006@fiu.edu')
dset.save(op.join(DATA_DIR, 'neurosynth.pkl.gz'))

# The NeuroVault 21-pain study NIDM-Results collection
utils.generate_pain_dset(DATA_DIR)

# Cognitive Atlas
annotate.ontology.cogat.download_cogat(out_dir=DATA_DIR, overwrite=True)

# ATHENA classifiers
annotate.ontology.cogpo.download_athena(out_dir=DATA_DIR, overwrite=True)

# PCC mask
from nilearn import datasets, image

atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
pcc_val = atlas.labels.index('Cingulate Gyrus, posterior division')
mask = image.math_img('img == {}'.format(pcc_val), img=atlas.maps)
mask.to_filename('posterior_cingulate_mask.nii.gz')
