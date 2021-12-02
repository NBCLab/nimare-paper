"""Build the GCLDA model to be used in the Jupyter Book."""

import os
from datetime import datetime

from nimare import annotate, dataset
from repo2data.repo2data import Repo2Data

start = datetime.now()

# Install the data if running locally, or points to cached data if running on neurolibre
DATA_REQ_FILE = os.path.join("../binder/data_requirement.json")
FIG_DIR = os.path.abspath("../images")

# Download data
repo2data = Repo2Data(DATA_REQ_FILE)
data_path = repo2data.install()
data_path = os.path.join(data_path[0], "data")

dataset_file = os.path.join(
    data_path,
    "neurosynth_dataset_first500_with_abstracts.pkl.gz",
)
neurosynth_dset_first_500 = dataset.Dataset.load(dataset_file)

counts_df = annotate.text.generate_counts(
    neurosynth_dset_first_500.texts,
    text_column="abstract",
    tfidf=False,
    min_df=10,
    max_df=0.95,
)

gclda_model = annotate.gclda.GCLDAModel(
    counts_df,
    neurosynth_dset_first_500.coordinates,
    n_regions=2,
    n_topics=50,
    symmetric=True,
    mask=neurosynth_dset_first_500.masker.mask_img,
)
gclda_model.fit(n_iters=2500, loglikely_freq=500)
gclda_model.save(os.path.join(data_path, "gclda_model.pkl.gz"))

end = datetime.now()
print(f"Model finished in {end - start}.")
