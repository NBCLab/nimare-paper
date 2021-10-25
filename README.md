# nimare-paper
The NiMARE software paper, as a Jupyter Book.

To build:

```bash
jupyter-book build content/
```

To view the built book, see https://nbclab.github.io/nimare-paper/

The book is configured to re-run all executable code if outputs are missing (`execute_notebooks` is set to `"auto"`).
If you want to build from scratch, then you can either change that setting in `content/_config.yml` or you can run `jupyter-book clean content/` before building.

## Notes

### Data files

The data are located at https://osf.io/t8h9c/.

### The amygdala mask

To create the amygdala mask, I did:

```python
from nilearn import datasets, image

atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
amyg_val = atlas["labels"].index("Right Amygdala")
amygdala_mask = image.math_img(f"img == {amyg_val}", img=atlas["maps"])
amygdala_mask.to_filename(os.path.join(DATA_DIR, "amygdala_roi.nii.gz"))
```

### `map_to_decode.nii.gz`

This is just the parameter estimate map from the DerSimonianLaird meta-analysis.

### Figures

While most of the figures in this manuscript are produced by the executed code, a few of them were manually created with Google Drawings.
Here are the links for those figures.

Figure 0: https://docs.google.com/drawings/d/1SMJL6x5UEkr6PjeKPXsh_qG1LXQaQj-ex1Dyjyi5LNY/edit

Figure 1: https://docs.google.com/drawings/d/1qhToDmOCbvpgpqQPH8RxGaOSox4BhKNlSM9hUdMsP-4/edit

Figure 2: https://docs.google.com/drawings/d/1u9xfy8KlThtiK8QuW0t9uyMu_DP32S9W6QcvurSFC2s/edit
