# nimare-paper
The NiMARE software paper, as a Jupyter Book.

To build:

```bash
jupyter-book build .
```

To view the built book, without any executed code or figures, see https://nbclab.github.io/nimare-paper/

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
