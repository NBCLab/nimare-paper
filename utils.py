import json
from glob import glob
import nibabel as nib
import pandas as pd
import numpy as np
from scipy import ndimage
from os.path import basename, join, isfile, dirname

import nimare


def _local_max(data, affine, min_distance):
    """Find all local maxima of the array, separated by at least min_distance.
    Adapted from https://stackoverflow.com/a/22631583/2589328
    Parameters
    ----------
    data : array_like
        3D array of with masked values for cluster.
    min_distance : :obj:`int`
        Minimum distance between local maxima in ``data``, in terms of mm.
    Returns
    -------
    ijk : :obj:`numpy.ndarray`
        (n_foci, 3) array of local maxima indices for cluster.
    vals : :obj:`numpy.ndarray`
        (n_foci,) array of values from data at ijk.
    """
    # Initial identification of subpeaks with minimal minimum distance
    data_max = ndimage.filters.maximum_filter(data, 3)
    maxima = (data == data_max)
    data_min = ndimage.filters.minimum_filter(data, 3)
    diff = ((data_max - data_min) > 0)
    maxima[diff == 0] = 0

    labeled, n_subpeaks = ndimage.label(maxima)
    ijk = np.array(ndimage.center_of_mass(data, labeled,
                                          range(1, n_subpeaks + 1)))
    ijk = np.round(ijk).astype(int)

    vals = np.apply_along_axis(arr=ijk, axis=1, func1d=_get_val,
                               input_arr=data)

    # Sort subpeaks in cluster in descending order of stat value
    order = (-vals).argsort()
    vals = vals[order]
    ijk = ijk[order, :]
    xyz = nib.affines.apply_affine(affine, ijk)  # Convert to xyz in mm

    # Reduce list of subpeaks based on distance
    keep_idx = np.ones(xyz.shape[0]).astype(bool)
    for i in range(xyz.shape[0]):
        for j in range(i + 1, xyz.shape[0]):
            if keep_idx[i] == 1:
                dist = np.linalg.norm(xyz[i, :] - xyz[j, :])
                keep_idx[j] = dist > min_distance
    ijk = ijk[keep_idx, :]
    vals = vals[keep_idx]
    return ijk, vals


def _get_val(row, input_arr):
    """Small function for extracting values from array based on index.
    """
    i, j, k = row
    return input_arr[i, j, k]


def _download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    req = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f_obj:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f_obj.write(chunk)
    return local_filename


def download_nidm_pain(out_dir):
    """
    Download NIDM Results for 21 pain studies from NeuroVault for tests.
    """
    url = 'https://neurovault.org/collections/1425/download'
    os.makedirs(out_dir, exist_ok=True)

    # Download
    fname = _download_file(url)

    # Unzip
    with zipfile.ZipFile(fname, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

    collection_folders = [f for f in glob(op.join(out_dir, '*')) if '.nidm' not in f]
    collection_folders = [f for f in collection_folders if op.isdir(f)]
    if len(collection_folders) > 1:
        raise Exception('More than one folder found: '
                        '{0}'.format(', '.join(collection_folders)))
    else:
        folder = collection_folders[0]
    zip_files = glob(op.join(folder, '*.zip'))
    for zf in zip_files:
        fn = op.splitext(op.basename(zf))[0]
        with zipfile.ZipFile(zf, 'r') as zip_ref:
            zip_ref.extractall(op.join(out_dir, fn))

    os.remove(fname)
    shutil.rmtree(folder)


def generate_pain_dset(out_dir):
    """
    Make a NiMARE dataset from the 21 pain study NeuroVault collection of NIDM
    Results packs.
    """
    nidmpack_dir = op.join(out_dir, 'pain_dset')
    download_nidm_pain(nidmpack_dir)
    json_file = op.join(nidmpack_dir, 'nidm_pain_dset.json')
    dset_dict = {}
    folders = sorted(glob(op.join(nidmpack_dir, 'pain_*.nidm')))
    for folder in folders:
        name = op.basename(folder)
        dset_dict[name] = {}
        dset_dict[name]['contrasts'] = {}
        dset_dict[name]['contrasts']['1'] = {}
        dset_dict[name]['contrasts']['1']['metadata'] = {}
        dset_dict[name]['contrasts']['1']['coords'] = {}
        dset_dict[name]['contrasts']['1']['coords']['space'] = 'MNI'
        dset_dict[name]['contrasts']['1']['images'] = {}
        dset_dict[name]['contrasts']['1']['images']['space'] = 'MNI_2mm'
        # con file
        files = glob(op.join(folder, 'Contrast*.nii.gz'))
        files = [f for f in files if 'StandardError' not in basename(f)]
        if files:
            f = sorted(files)[0]
        else:
            f = None
        dset_dict[name]['contrasts']['1']['images']['con'] = f
        # se file
        files = glob(op.join(folder, 'ContrastStandardError*.nii.gz'))
        if files:
            f = sorted(files)[0]
        else:
            f = None
        dset_dict[name]['contrasts']['1']['images']['se'] = f
        # z file
        files = glob(op.join(folder, 'ZStatistic*.nii.gz'))
        if files:
            f = sorted(files)[0]
        else:
            f = None
        dset_dict[name]['contrasts']['1']['images']['z'] = f
        # t file
        files = glob(op.join(folder, 'TStatistic*.nii.gz'))
        if files:
            f = sorted(files)[0]
        else:
            f = None
        dset_dict[name]['contrasts']['1']['images']['t'] = f
        # sample size
        f = op.join(folder, 'DesignMatrix.csv')
        if op.isfile(f):
            df = pd.read_csv(f, header=None)
            n = [df.shape[0]]
        else:
            n = None
        dset_dict[name]['contrasts']['1']['metadata']['sample_sizes'] = n
        # foci
        files = glob(op.join(folder, 'ExcursionSet*.nii.gz'))
        f = sorted(files)[0]
        img = nib.load(f)
        data = np.nan_to_num(img.get_data())
        # positive clusters
        binarized = np.copy(data)
        binarized[binarized > 0] = 1
        binarized[binarized < 0] = 0
        binarized = binarized.astype(int)
        labeled = ndimage.measurements.label(binarized, np.ones((3, 3, 3)))[0]
        clust_ids = sorted(list(np.unique(labeled)[1:]))

        peak_vals = np.array([np.max(data * (labeled == c)) for c in clust_ids])
        clust_ids = [clust_ids[c] for c in (-peak_vals).argsort()]  # Sort by descending max value

        ijk = []
        for c_id, c_val in enumerate(clust_ids):
            cluster_mask = labeled == c_val
            masked_data = data * cluster_mask

            # Get peaks, subpeaks and associated statistics
            subpeak_ijk, subpeak_vals = _local_max(masked_data, img.affine, min_distance=8)

            # Only report peak and, at most, top 3 subpeaks.
            n_subpeaks = np.min((len(subpeak_vals), 4))
            subpeak_ijk = subpeak_ijk[:n_subpeaks, :]
            ijk.append(subpeak_ijk)
        ijk = np.vstack(ijk)
        xyz = nib.affines.apply_affine(img.affine, ijk)
        dset_dict[name]['contrasts']['1']['coords']['x'] = list(xyz[:, 0])
        dset_dict[name]['contrasts']['1']['coords']['y'] = list(xyz[:, 1])
        dset_dict[name]['contrasts']['1']['coords']['z'] = list(xyz[:, 2])

    with open(json_file, 'w') as fo:
        json.dump(dset_dict, fo, sort_keys=True, indent=4)

    dset = nimare.dataset.Dataset(json_file)
    dset.save(op.join(out_dir, 'pain.pkl.gz'))
