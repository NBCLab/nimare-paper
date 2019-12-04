import nimare

dset = nimare.dataset.Dataset.load(op.join(DATADIR, 'neurosynth.pkl.gz'))
pain_ids = dset.get_studies_by_label('pain')
not_pain_ids = sorted(list(set(dset.ids) - set(pain_ids)))
pain_dset = dset.slice(pain_ids)
not_pain_dset = dset.slice(not_pain_ids)

# Now we can run a bunch of coordinate-based meta-analyses
