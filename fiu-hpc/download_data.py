"""A script to be called before building the book on the FIU HPC.

FIU HPC processing nodes do not have internet access, so they must have the data downloaded before
the build job is started.
I activate the NiMARE miniconda environment and then run this script in a login node with internet
access.
"""
import os

from repo2data.repo2data import Repo2Data

# Install the data if running locally, or points to cached data if running on neurolibre
DATA_REQ_FILE = os.path.abspath("../binder/data_requirement.json")
repo2data = Repo2Data(DATA_REQ_FILE)
data_path = repo2data.install()
