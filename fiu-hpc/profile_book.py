"""Execute each chapter in the Jupyter Book and measure memory usage.

I need to run `jupytext *.md --to ipynb` in ../content/ before this.
"""
import os
from glob import glob

import nbformat
from memory_profiler import memory_usage
from nbclient import NotebookClient


def run_page(filename):
    nb = nbformat.read(filename, as_version=4)
    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": "."}},
    )
    client.execute()


os.chdir("/home/data/nbc/misc-projects/Salo_NiMARE/content/")

chapters = sorted(glob("/home/data/nbc/misc-projects/Salo_NiMARE/content/*.ipynb"))
for chapter in chapters:
    print(os.path.basename(chapter))
    chapter_file = os.path.abspath(chapter)

    mem_usage = memory_usage((run_page, (chapter_file,)), interval=0.5, timeout=None)
    print(f"Memory usage (in chunks of 0.5 seconds): {mem_usage}")
    print(f"Maximum memory usage: {max(mem_usage)} MB")
