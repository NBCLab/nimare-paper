# nimare-paper
The NiMARE software paper, as a Jupyter Book.

To view the built book, see https://nbclab.github.io/nimare-paper/

[![status](http://neurolibre.herokuapp.com/papers/28dfe9bf9747b20c7f70221badb19baf/status.svg)](http://neurolibre.herokuapp.com/papers/28dfe9bf9747b20c7f70221badb19baf)

## Building the book locally

### 1. Install dependencies

In order to execute the book's code, you will need install all of the Python libraries that are required.
The necessary requirements and associated versions are available in `binder/requirements.txt`.

You can install them with the following:

```
pip install binder/requirements.txt
```

In addition to the Python requirements, the LDA topic model section requires Java, in order to use MALLET.

### 2. Download data files

The data files necessary to execute the code in this book are located at https://drive.google.com/uc?id=1LgscyPqnka163hu5mdJ1X7UvX3IvXDJ2 in a zip file.
You can either download these files to a `data/` folder at the same level as `content/`, _or_ you can rely on `repo2data` to download the files automatically during the book build.

### 3. Build the book

To build:

```bash
jupyter-book build content/
```

The book is configured to rely on the pre-generated cache (`execute_notebooks` is set to `"cache"`).
If you want to build from scratch, then you can either change that setting in `content/_config.yml` or you can run `jupyter-book clean content/` before building.

## Notes

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
