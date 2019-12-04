from nimare import annotate

DATA_DIR = 'raw_data'
MODEL_DIR = 'models'

# Simple upstream weights. No floats allowed.
WEIGHTS = {'isKindOf': 1, 'isPartOf': 1, 'inCategory': 1}

dset = nimare.dataset.Dataset.load(op.join(DATADIR, 'neurosynth.pkl.gz'))
cogat_aliases, cogat_relationships = annotate.ontology.cogat.download_cogat(
    out_dir=DATA_DIR, overwrite=False)
counts_df, _ = annotate.ontology.cogat.extract_cogat(
    dset.texts, cogat_aliases, text_column='abstract')
expanded_df = annotate.ontology.cogat.expand_counts(counts_df, cogat_relationships, WEIGHTS)

coordinates_df = dset.coordinates
gclda = annotate.topic.GCLDAModel(
    expanded_df, coordinates_df,
    mask='mni152_2mm', n_topics=100, n_regions=2, symmetric=True,
    alpha=0.1, beta=0.01, gamma=0.01, delta=1.0,
    dobs=25, roi_size=50.0, seed_init=1)
gclda.fit(n_iters=10000, loglikely_freq=100)
gclda.save(op.join(MODEL_DIR, 'gclda.pkl.gz'))

lda = LDAModel(
    dset.texts, text_column='abstract', n_topics=100,
    n_iters=10000, alpha='auto', beta='0.001')
lda.fit()
lda.save(op.join(MODEL_DIR, 'lda.pkl.gz'))
