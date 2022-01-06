#!/usr/bin/env python
# coding: utf-8

# # Download the Data

# In[1]:


# First, import the necessary modules and functions
import os

from repo2data.repo2data import Repo2Data

# Install the data if running locally, or points to cached data if running on neurolibre
DATA_REQ_FILE = os.path.abspath("../binder/data_requirement.json")
repo2data = Repo2Data(DATA_REQ_FILE)
data_path = repo2data.install()
data_path = os.path.join(data_path[0], "data")
print(f"Data are located at {data_path}")

