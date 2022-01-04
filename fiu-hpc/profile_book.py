"""Execute each chapter in the Jupyter Book and measure memory usage."""
import os
from glob import glob

import nbformat
from memory_profiler import memory_usage
from nbclient import NotebookClient


def run_page(filename):
    nb = nbformat.read(filename, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": "notebooks/"}},
    )
    client.execute()


chapters = sorted(glob("../contents/*.md"))
for chapter in chapters:
    print(os.path.basename(chapter))
    chapter_file = os.path.abspath(chapter)

    mem_usage = memory_usage((run_page, chapter_file))
    print(f"Memory usage (in chunks of .1 seconds): {mem_usage}")
    print(f"Maximum memory usage: {max(mem_usage)}")
