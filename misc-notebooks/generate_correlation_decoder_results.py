"""Run a CorrelationDecoder to be used in the Jupyter book."""

import os
from datetime import datetime

import nimare
from nimare import decode, meta
from repo2data.repo2data import Repo2Data

start = datetime.now()

# Install the data if running locally, or points to cached data if running on neurolibre
DATA_REQ_FILE = os.path.join("../binder/data_requirement.json")

# Download data
repo2data = Repo2Data(DATA_REQ_FILE)
data_path = repo2data.install()
data_path = os.path.join(data_path[0], "data")

# Load the appropriate dataset
dset_file = os.path.join(data_path, "neurosynth_dataset_first500_with_mkda_ma.pkl.gz")
kern = meta.kernel.MKDAKernel(memory_limit=None)
target_folder = os.path.join(data_path, "neurosynth_dataset_maps")
if not os.path.isfile(dset_file):
    neurosynth_dset_first500 = nimare.dataset.Dataset.load(
        os.path.join(data_path, "neurosynth_dataset_first500.pkl.gz")
    )
    os.makedirs(target_folder, exist_ok=True)
    neurosynth_dset_first500.update_path(target_folder)
    neurosynth_dset_first500 = kern.transform(neurosynth_dset_first500, return_type="dataset")
    neurosynth_dset_first500.save(dset_file)
else:
    neurosynth_dset_first500 = nimare.dataset.Dataset.load(dset_file)
    neurosynth_dset_first500.update_path(target_folder)

# Collect features for decoding
# We use any features that appear in >10% of studies and <90%.
id_cols = ["id", "study_id", "contrast_id"]
frequency_threshold = 0.001
cols = neurosynth_dset_first500.annotations.columns
cols = [c for c in cols if c not in id_cols]
df = neurosynth_dset_first500.annotations.copy()[cols]
n_studies = df.shape[0]
feature_counts = (df >= frequency_threshold).sum(axis=0)
target_features = feature_counts.between(n_studies * 0.1, n_studies * 0.9)
target_features = target_features[target_features]
target_features = target_features.index.values
print(f"{len(target_features)} features selected.", flush=True)

continuous_map = os.path.join(data_path, "map_to_decode.nii.gz")

# Here we train the decoder
corr_decoder = decode.continuous.CorrelationDecoder(
    frequency_threshold=0.001,
    meta_estimator=meta.MKDADensity(kernel_transformer=kern, memory_limit=None),
    target_image="z",
    features=target_features,
    memory_limit="500mb",
)
corr_decoder.fit(neurosynth_dset_first500)
corr_df = corr_decoder.transform(continuous_map)
corr_df.to_csv(
    os.path.join(data_path, "correlation_decoder_results.tsv"),
    sep="\t",
    index=True,
    index_label="feature",
)
