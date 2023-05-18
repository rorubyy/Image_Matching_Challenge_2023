# General utilities
import os
from tqdm import tqdm
from time import time
from fastprogress import progress_bar
import gc
import numpy as np
import h5py
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy

# CV/ML
import cv2
import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# 3D reconstruction
import pycolmap

print('Kornia version', K.__version__)
print('Pycolmap version', pycolmap.__version__)

LOCAL_FEATURE = 'KeyNetAffNetHardNet'
device = torch.device('cuda')
# Can be LoFTR, KeyNetAffNetHardNet, or DISK

# def arr_to_str(a):
#     return ';'.join([str(x) for x in a.reshape(-1)])


def load_torch_image(fname, device=torch.device('cpu')):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img.to(device))
    return img

# We will use ViT global descriptor to get matching shortlists.


def get_global_desc(fnames, model,
                    device=torch.device('cpu')):
    model = model.eval()
    model = model.to(device)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    global_descs_convnext = []
    for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]
        img = Image.open(img_fname_full).convert('RGB')
        timg = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            desc = model.forward_features(timg.to(device)).mean(dim=(-1, 2))
            #print (desc.shape)
            desc = desc.view(1, -1)
            desc_norm = F.normalize(desc, dim=1, p=2)
        #print (desc_norm)
        global_descs_convnext.append(desc_norm.detach().cpu())
    global_descs_all = torch.cat(global_descs_convnext, dim=0)
    return global_descs_all


def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i+1, len(img_fnames)):
            index_pairs.append((i, j))
    return index_pairs


def get_image_pairs_shortlist(fnames,
                              sim_th=0.6,  # should be strict
                              min_pairs=20,
                              exhaustive_if_less=20,
                              device=torch.device('cpu')):
    num_imgs = len(fnames)

    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)
    model = torch.load(
        '/root/code/3Dreconstruction/model/swinv2_base_patch4_window12_192_22k.pth')
    model = timm.create_model('tf_efficientnet_b7',
                              checkpoint_path='/root/code/3Dreconstruction/model/tf_efficientnet_b7_ra-6c08e654.pth')
    # model = timm.create_model('tf_efficientnet_b7',
    #                           checkpoint_path='/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b7/1/tf_efficientnet_b7_ra-6c08e654.pth')
    model.eval()
    descs = get_global_desc(fnames, model, device=device)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    # removing half
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    already_there_set = []
    for st_idx in range(num_imgs-1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total += 1
    matching_list = sorted(list(set(matching_list)))
    return matching_list

# import sys
# import sqlite3
# import numpy as np


# IS_PYTHON3 = sys.version_info[0] >= 3

# MAX_IMAGE_ID = 2**31 - 1

# CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
#     camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
#     model INTEGER NOT NULL,
#     width INTEGER NOT NULL,
#     height INTEGER NOT NULL,
#     params BLOB,
#     prior_focal_length INTEGER NOT NULL)"""

# CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
#     image_id INTEGER PRIMARY KEY NOT NULL,
#     rows INTEGER NOT NULL,
#     cols INTEGER NOT NULL,
#     data BLOB,
#     FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

# CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
#     image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
#     name TEXT NOT NULL UNIQUE,
#     camera_id INTEGER NOT NULL,
#     prior_qw REAL,
#     prior_qx REAL,
#     prior_qy REAL,
#     prior_qz REAL,
#     prior_tx REAL,
#     prior_ty REAL,
#     prior_tz REAL,
#     CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
#     FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
# """.format(MAX_IMAGE_ID)

# CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
# CREATE TABLE IF NOT EXISTS two_view_geometries (
#     pair_id INTEGER PRIMARY KEY NOT NULL,
#     rows INTEGER NOT NULL,
#     cols INTEGER NOT NULL,
#     data BLOB,
#     config INTEGER NOT NULL,
#     F BLOB,
#     E BLOB,
#     H BLOB)
# """

# CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
#     image_id INTEGER PRIMARY KEY NOT NULL,
#     rows INTEGER NOT NULL,
#     cols INTEGER NOT NULL,
#     data BLOB,
#     FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
# """

# CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
#     pair_id INTEGER PRIMARY KEY NOT NULL,
#     rows INTEGER NOT NULL,
#     cols INTEGER NOT NULL,
#     data BLOB)"""

# CREATE_NAME_INDEX = \
#     "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

# CREATE_ALL = "; ".join([
#     CREATE_CAMERAS_TABLE,
#     CREATE_IMAGES_TABLE,
#     CREATE_KEYPOINTS_TABLE,
#     CREATE_DESCRIPTORS_TABLE,
#     CREATE_MATCHES_TABLE,
#     CREATE_TWO_VIEW_GEOMETRIES_TABLE,
#     CREATE_NAME_INDEX
# ])


# def image_ids_to_pair_id(image_id1, image_id2):
#     if image_id1 > image_id2:
#         image_id1, image_id2 = image_id2, image_id1
#     return image_id1 * MAX_IMAGE_ID + image_id2


# def pair_id_to_image_ids(pair_id):
#     image_id2 = pair_id % MAX_IMAGE_ID
#     image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
#     return image_id1, image_id2


# def array_to_blob(array):
#     if IS_PYTHON3:
#         return array.tostring()
#     else:
#         return np.getbuffer(array)


# def blob_to_array(blob, dtype, shape=(-1,)):
#     if IS_PYTHON3:
#         return np.fromstring(blob, dtype=dtype).reshape(*shape)
#     else:
#         return np.frombuffer(blob, dtype=dtype).reshape(*shape)


# class COLMAPDatabase(sqlite3.Connection):

#     @staticmethod
#     def connect(database_path):
#         return sqlite3.connect(database_path, factory=COLMAPDatabase)


#     def __init__(self, *args, **kwargs):
#         super(COLMAPDatabase, self).__init__(*args, **kwargs)

#         self.create_tables = lambda: self.executescript(CREATE_ALL)
#         self.create_cameras_table = \
#             lambda: self.executescript(CREATE_CAMERAS_TABLE)
#         self.create_descriptors_table = \
#             lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
#         self.create_images_table = \
#             lambda: self.executescript(CREATE_IMAGES_TABLE)
#         self.create_two_view_geometries_table = \
#             lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
#         self.create_keypoints_table = \
#             lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
#         self.create_matches_table = \
#             lambda: self.executescript(CREATE_MATCHES_TABLE)
#         self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

#     def add_camera(self, model, width, height, params,
#                    prior_focal_length=False, camera_id=None):
#         params = np.asarray(params, np.float64)
#         cursor = self.execute(
#             "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
#             (camera_id, model, width, height, array_to_blob(params),
#              prior_focal_length))
#         return cursor.lastrowid

#     def add_image(self, name, camera_id,
#                   prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
#         cursor = self.execute(
#             "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
#             (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
#              prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
#         return cursor.lastrowid

#     def add_keypoints(self, image_id, keypoints):
#         assert(len(keypoints.shape) == 2)
#         assert(keypoints.shape[1] in [2, 4, 6])

#         keypoints = np.asarray(keypoints, np.float32)
#         self.execute(
#             "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
#             (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

#     def add_descriptors(self, image_id, descriptors):
#         descriptors = np.ascontiguousarray(descriptors, np.uint8)
#         self.execute(
#             "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
#             (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

#     def add_matches(self, image_id1, image_id2, matches):
#         assert(len(matches.shape) == 2)
#         assert(matches.shape[1] == 2)

#         if image_id1 > image_id2:
#             matches = matches[:,::-1]

#         pair_id = image_ids_to_pair_id(image_id1, image_id2)
#         matches = np.asarray(matches, np.uint32)
#         self.execute(
#             "INSERT INTO matches VALUES (?, ?, ?, ?)",
#             (pair_id,) + matches.shape + (array_to_blob(matches),))

#     def add_two_view_geometry(self, image_id1, image_id2, matches,
#                               F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
#         assert(len(matches.shape) == 2)
#         assert(matches.shape[1] == 2)

#         if image_id1 > image_id2:
#             matches = matches[:,::-1]

#         pair_id = image_ids_to_pair_id(image_id1, image_id2)
#         matches = np.asarray(matches, np.uint32)
#         F = np.asarray(F, dtype=np.float64)
#         E = np.asarray(E, dtype=np.float64)
#         H = np.asarray(H, dtype=np.float64)
#         self.execute(
#             "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
#             (pair_id,) + matches.shape + (array_to_blob(matches), config,
#              array_to_blob(F), array_to_blob(E), array_to_blob(H)))


# import os, argparse, h5py, warnings
# import numpy as np
# from tqdm import tqdm
# from PIL import Image, ExifTags


# def get_focal(image_path, err_on_default=False):
#     image         = Image.open(image_path)
#     max_size      = max(image.size)

#     exif = image.getexif()
#     focal = None
#     if exif is not None:
#         focal_35mm = None
#         # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
#         for tag, value in exif.items():
#             focal_35mm = None
#             if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
#                 focal_35mm = float(value)
#                 break

#         if focal_35mm is not None:
#             focal = focal_35mm / 35. * max_size

#     if focal is None:
#         if err_on_default:
#             raise RuntimeError("Failed to find focal length")

#         # failed to find it in exif, use prior
#         FOCAL_PRIOR = 1.2
#         focal = FOCAL_PRIOR * max_size

#     return focal

# def create_camera(db, image_path, camera_model):
#     image         = Image.open(image_path)
#     width, height = image.size

#     focal = get_focal(image_path)

#     if camera_model == 'simple-pinhole':
#         model = 0 # simple pinhole
#         param_arr = np.array([focal, width / 2, height / 2])
#     if camera_model == 'pinhole':
#         model = 1 # pinhole
#         param_arr = np.array([focal, focal, width / 2, height / 2])
#     elif camera_model == 'simple-radial':
#         model = 2 # simple radial
#         param_arr = np.array([focal, width / 2, height / 2, 0.1])
#     elif camera_model == 'opencv':
#         model = 4 # opencv
#         param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])

#     return db.add_camera(model, width, height, param_arr)


# def add_keypoints(db, h5_path, image_path, img_ext, camera_model, single_camera = True):
#     keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')

#     camera_id = None
#     fname_to_id = {}
#     for filename in tqdm(list(keypoint_f.keys())):
#         keypoints = keypoint_f[filename][()]

#         fname_with_ext = filename# + img_ext
#         path = os.path.join(image_path, fname_with_ext)
#         if not os.path.isfile(path):
#             raise IOError(f'Invalid image path {path}')

#         if camera_id is None or not single_camera:
#             camera_id = create_camera(db, path, camera_model)
#         image_id = db.add_image(fname_with_ext, camera_id)
#         fname_to_id[filename] = image_id

#         db.add_keypoints(image_id, keypoints)

#     return fname_to_id

# def add_matches(db, h5_path, fname_to_id):
#     match_file = h5py.File(os.path.join(h5_path, 'matches.h5'), 'r')

#     added = set()
#     n_keys = len(match_file.keys())
#     n_total = (n_keys * (n_keys - 1)) // 2

#     with tqdm(total=n_total) as pbar:
#         for key_1 in match_file.keys():
#             group = match_file[key_1]
#             for key_2 in group.keys():
#                 id_1 = fname_to_id[key_1]
#                 id_2 = fname_to_id[key_2]

#                 pair_id = image_ids_to_pair_id(id_1, id_2)
#                 if pair_id in added:
#                     warnings.warn(f'Pair {pair_id} ({id_1}, {id_2}) already added!')
#                     continue

#                 matches = group[key_2][()]
#                 db.add_matches(id_1, id_2, matches)

#                 added.add(pair_id)

#                 pbar.update(1)

# # Making kornia local features loading w/o internet
# class KeyNetAffNetHardNet(KF.LocalFeature):
#     """Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor.

#     .. image:: _static/img/keynet_affnet.jpg
#     """

#     def __init__(
#         self,
#         num_features: int = 5000,
#         upright: bool = False,
#         device = torch.device('cpu'),
#         scale_laf: float = 1.0,
#     ):
#         ori_module = KF.PassLAF() if upright else KF.LAFOrienter(angle_detector=KF.OriNet(False)).eval()
#         if not upright:
#             weights = torch.load('/kaggle/input/kornia-local-feature-weights/OriNet.pth')['state_dict']
#             ori_module.angle_detector.load_state_dict(weights)
#         detector = KF.KeyNetDetector(
#             False, num_features=num_features, ori_module=ori_module, aff_module=KF.LAFAffNetShapeEstimator(False).eval()
#         ).to(device)
#         kn_weights = torch.load('/kaggle/input/kornia-local-feature-weights/keynet_pytorch.pth')['state_dict']
#         detector.model.load_state_dict(kn_weights)
#         affnet_weights = torch.load('/kaggle/input/kornia-local-feature-weights/AffNet.pth')['state_dict']
#         detector.aff.load_state_dict(affnet_weights)

#         hardnet = KF.HardNet(False).eval()
#         hn_weights = torch.load('/kaggle/input/kornia-local-feature-weights/HardNetLib.pth')['state_dict']
#         hardnet.load_state_dict(hn_weights)
#         descriptor = KF.LAFDescriptor(hardnet, patch_size=32, grayscale_descriptor=True).to(device)
#         super().__init__(detector, descriptor, scale_laf)


# def detect_features(img_fnames,
#                     num_feats = 2048,
#                     upright = False,
#                     device=torch.device('cpu'),
#                     feature_dir = '.featureout',
#                     resize_small_edge_to = 600):
#     if LOCAL_FEATURE == 'DISK':
#         # Load DISK from Kaggle models so it can run when the notebook is offline.
#         disk = KF.DISK().to(device)
#         pretrained_dict = torch.load('/kaggle/input/disk/pytorch/depth-supervision/1/loftr_outdoor.ckpt', map_location=device)
#         disk.load_state_dict(pretrained_dict['extractor'])
#         disk.eval()
#     if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
#         feature = KeyNetAffNetHardNet(num_feats, upright, device).to(device).eval()
#     if not os.path.isdir(feature_dir):
#         os.makedirs(feature_dir)
#     with h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
#          h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
#          h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
#         for img_path in progress_bar(img_fnames):
#             img_fname = img_path.split('/')[-1]
#             key = img_fname
#             with torch.inference_mode():
#                 timg = load_torch_image(img_path, device=device)
#                 H, W = timg.shape[2:]
#                 if resize_small_edge_to is None:
#                     timg_resized = timg
#                 else:
#                     timg_resized = K.geometry.resize(timg, resize_small_edge_to, antialias=True)
#                     print(f'Resized {timg.shape} to {timg_resized.shape} (resize_small_edge_to={resize_small_edge_to})')
#                 h, w = timg_resized.shape[2:]
#                 if LOCAL_FEATURE == 'DISK':
#                     features = disk(timg_resized, num_feats, pad_if_not_divisible=True)[0]
#                     kps1, descs = features.keypoints, features.descriptors

#                     lafs = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
#                 if LOCAL_FEATURE == 'KeyNetAffNetHardNet':
#                     lafs, resps, descs = feature(K.color.rgb_to_grayscale(timg_resized))
#                 lafs[:,:,0,:] *= float(W) / float(w)
#                 lafs[:,:,1,:] *= float(H) / float(h)
#                 desc_dim = descs.shape[-1]
#                 kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
#                 descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
#                 f_laf[key] = lafs.detach().cpu().numpy()
#                 f_kp[key] = kpts
#                 f_desc[key] = descs
#     return

# def get_unique_idxs(A, dim=0):
#     # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
#     unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
#     _, ind_sorted = torch.sort(idx, stable=True)
#     cum_sum = counts.cumsum(0)
#     cum_sum = torch.cat((torch.tensor([0],device=cum_sum.device), cum_sum[:-1]))
#     first_indices = ind_sorted[cum_sum]
#     return first_indices

# def match_features(img_fnames,
#                    index_pairs,
#                    feature_dir = '.featureout',
#                    device=torch.device('cpu'),
#                    min_matches=15,
#                    force_mutual = True,
#                    matching_alg='smnn'
#                   ):
#     assert matching_alg in ['smnn', 'adalam']
#     with h5py.File(f'{feature_dir}/lafs.h5', mode='r') as f_laf, \
#          h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
#         h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:

#         for pair_idx in progress_bar(index_pairs):
#                     idx1, idx2 = pair_idx
#                     fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
#                     key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
#                     lafs1 = torch.from_numpy(f_laf[key1][...]).to(device)
#                     lafs2 = torch.from_numpy(f_laf[key2][...]).to(device)
#                     desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
#                     desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
#                     if matching_alg == 'adalam':
#                         img1, img2 = cv2.imread(fname1), cv2.imread(fname2)
#                         hw1, hw2 = img1.shape[:2], img2.shape[:2]
#                         adalam_config = KF.adalam.get_adalam_default_config()
#                         #adalam_config['orientation_difference_threshold'] = None
#                         #adalam_config['scale_rate_threshold'] = None
#                         adalam_config['force_seed_mnn']= False
#                         adalam_config['search_expansion'] = 16
#                         adalam_config['ransac_iters'] = 128
#                         adalam_config['device'] = device
#                         dists, idxs = KF.match_adalam(desc1, desc2,
#                                                       lafs1, lafs2, # Adalam takes into account also geometric information
#                                                       hw1=hw1, hw2=hw2,
#                                                       config=adalam_config) # Adalam also benefits from knowing image size
#                     else:
#                         dists, idxs = KF.match_smnn(desc1, desc2, 0.98)
#                     if len(idxs)  == 0:
#                         continue
#                     # Force mutual nearest neighbors
#                     if force_mutual:
#                         first_indices = get_unique_idxs(idxs[:,1])
#                         idxs = idxs[first_indices]
#                         dists = dists[first_indices]
#                     n_matches = len(idxs)
#                     if False:
#                         print (f'{key1}-{key2}: {n_matches} matches')
#                     group  = f_match.require_group(key1)
#                     if n_matches >= min_matches:
#                          group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
#     return


def resize_img_loftr(img, max_len, enlarge_scale, variant_scale, device):
    if max_len == -1:
        scale = 1
    else:
        scale = max(max_len, max(img.shape[0], img.shape[1]) * enlarge_scale) / max(img.shape[0], img.shape[1])
    w = int(round(img.shape[1] * scale) / 8) * 8
    h = int(round(img.shape[0] * scale) / 8) * 8
    
    isResized = False
    if w >= h:
        if int(h * variant_scale) <= w:
            isResized = True
            h = int(h * variant_scale / 8) * 8
    else:
        if int(w * variant_scale) <= h:
            isResized = True
            w = int(w * variant_scale / 8) * 8
    img_resize = cv2.resize(img, (w, h)) 
    img_resize = K.image_to_tensor(img_resize, False).float() / 255.
    
    return img_resize.to(device), (w / img.shape[1], h / img.shape[0]), isResized


def matcher(img_fnames,
            index_pairs,
            feature_dir='.featureout_loftr',
            device=torch.device('cpu'),
            min_matches=15, resize_to_=(640, 480)):
    
    scales_lens_loftr = [[1.1, 1000, 1.0], [1, 1200, 1.3], [0.9, 1400, 1.6]]
    scales_lens_superglue = [[1.2, 1200, 1.0], [1.2, 1600, 1.6], [0.8, 2000, 2], [1, 2800, 3]]
    w_h_muts_dkm = [[680 * 510, 1]]

    matcher_loftr = KF.LoFTR(pretrained=None)
    matcher_loftr.load_state_dict(torch.load(
        '/root/code/3Dreconstruction/kornia-loftr/loftr_outdoor.ckpt')['state_dict'])
    matcher_loftr = matcher_loftr.to(device).eval()

    for pair_idx in progress_bar(index_pairs):
        idx1, idx2 = pair_idx
        fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
        key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
        image_0_BGR = cv2.imread(fname1)
        image_1_BGR = cv2.imread(fname2)
        
        image_0_GRAY = cv2.cvtColor(image_0_BGR, cv2.COLOR_BGR2GRAY)
        image_1_GRAY = cv2.cvtColor(image_1_BGR, cv2.COLOR_BGR2GRAY)
        
        # ----- LoFTR -----        
        mkpts0_loftr_all = []
        mkpts1_loftr_all = []
        for variant_scale, max_len, enlarge_scale in scales_lens_loftr:
            
            image_0_resize, scale_0, isResized_0 = resize_img_loftr(image_0_GRAY, max_len, enlarge_scale, variant_scale, device)
            image_1_resize, scale_1, isResized_1 = resize_img_loftr(image_1_GRAY, max_len, enlarge_scale, variant_scale, device)
            
            if isResized_0 == False or isResized_1 == False: continue
            
            input_dict = {"image0": image_0_resize, 
                      "image1": image_1_resize}
            correspondences = matcher_loftr(input_dict)
            confidence = correspondences['confidence'].cpu().numpy()
            
            if len(confidence) < 1: continue

            confidence_quantile = np.quantile(confidence, 0.6)
            idx = np.where(confidence >= confidence_quantile)
            
            mkpts0_loftr = correspondences['keypoints0'].cpu().numpy()[idx]
            mkpts1_loftr = correspondences['keypoints1'].cpu().numpy()[idx]
            
            print("loftr scale_0", scale_0)
            print("loftr scale_1", scale_1)

            mkpts0_loftr = mkpts0_loftr / scale_0
            mkpts1_loftr = mkpts1_loftr / scale_1

            mkpts0_loftr_all.append(mkpts0_loftr)
            mkpts1_loftr_all.append(mkpts1_loftr)
        
        mkpts0_loftr_all = np.concatenate(mkpts0_loftr_all, axis=0)
        mkpts1_loftr_all = np.concatenate(mkpts1_loftr_all, axis=0) 
        
    return


def match_loftr(img_fnames,
                index_pairs,
                feature_dir='.featureout_loftr',
                device=torch.device('cpu'),
                min_matches=15, resize_to_=(640, 480)):
    matcher = KF.LoFTR(pretrained=None)
    matcher.load_state_dict(torch.load(
        '/root/code/3Dreconstruction/kornia-loftr/loftr_outdoor.ckpt')['state_dict'])
    matcher = matcher.to(device).eval()

    # First we do pairwise matching, and then extract "keypoints" from loftr matches.
    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='w') as f_match:
        for pair_idx in progress_bar(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
            # Load img1
            timg1 = K.color.rgb_to_grayscale(
                load_torch_image(fname1, device=device))
            H1, W1 = timg1.shape[2:]
            if H1 < W1:
                resize_to = resize_to_[1], resize_to_[0]
            else:
                resize_to = resize_to_
            timg_resized1 = K.geometry.resize(timg1, resize_to, antialias=True)
            h1, w1 = timg_resized1.shape[2:]

            # Load img2
            timg2 = K.color.rgb_to_grayscale(
                load_torch_image(fname2, device=device))
            H2, W2 = timg2.shape[2:]
            if H2 < W2:
                resize_to2 = resize_to[1], resize_to[0]
            else:
                resize_to2 = resize_to_
            timg_resized2 = K.geometry.resize(
                timg2, resize_to2, antialias=True)
            h2, w2 = timg_resized2.shape[2:]
            with torch.inference_mode():
                input_dict = {"image0": timg_resized1, "image1": timg_resized2}
                correspondences = matcher(input_dict)
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()

            mkpts0[:, 0] *= float(W1) / float(w1)
            mkpts0[:, 1] *= float(H1) / float(h1)

            mkpts1[:, 0] *= float(W2) / float(w2)
            mkpts1[:, 1] *= float(H2) / float(h2)

            n_matches = len(mkpts1)
            group = f_match.require_group(key1)
            if n_matches >= min_matches:
                group.create_dataset(
                    key2, data=np.concatenate([mkpts0, mkpts1], axis=1))

    # Let's find unique loftr pixels and group them together.
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts = defaultdict(int)
    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='r') as f_match:
        for k1 in f_match.keys():
            group = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(
                    len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0] += total_kpts[k1]
                current_match[:, 1] += total_kpts[k2]
                total_kpts[k1] += len(matches)
                total_kpts[k2] += len(matches)
                match_indexes[k1][k2] = current_match

    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(
            torch.from_numpy(kpts[k]), dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
            m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
            mkpts = np.concatenate([unique_kpts[k1][m2[:, 0]],
                                    unique_kpts[k2][m2[:, 1]],
                                    ],
                                   axis=1)
            unique_idxs_current = get_unique_idxs(
                torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1

    with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for k1, gr in out_match.items():
            group = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match
    return

# def import_into_colmap(img_dir,
#                        feature_dir ='.featureout',
#                        database_path = 'colmap.db',
#                        img_ext='.jpg'):
#     db = COLMAPDatabase.connect(database_path)
#     db.create_tables()
#     single_camera = False
#     fname_to_id = add_keypoints(db, feature_dir, img_dir, img_ext, 'simple-radial', single_camera)
#     add_matches(
#         db,
#         feature_dir,
#         fname_to_id,
#     )

#     db.commit()
#     return


src = '/root/code/3Dreconstruction/kaggle/input/image-matching-challenge-2023'
# Get train data
data_dict = {}

with open(f'{src}/train/train_labels.csv', 'r') as f:
    for i, l in enumerate(f):
        # Skip header.
        if l and i > 0:
            dataset, scene, image, _, _ = l.strip().split(',')
            if dataset not in data_dict:
                data_dict[dataset] = {}
            if scene not in data_dict[dataset]:
                data_dict[dataset][scene] = []
            data_dict[dataset][scene].append(image)

# Get data from csv.

# data_dict = {}
# with open(f'{src}/sample_submission.csv', 'r') as f:
#     for i, l in enumerate(f):
#         # Skip header.
#         if l and i > 0:
#             image, dataset, scene, _, _ = l.strip().split(',')
#             if dataset not in data_dict:
#                 data_dict[dataset] = {}
#             if scene not in data_dict[dataset]:
#                 data_dict[dataset][scene] = []
#             data_dict[dataset][scene].append(image)

for dataset in data_dict:
    for scene in data_dict[dataset]:
        print(
            f'{dataset} / {scene} -> {len(data_dict[dataset][scene])} images')

out_results = {}
timings = {"shortlisting": [],
           "feature_detection": [],
           "feature_matching": [],
           "RANSAC": [],
           "Reconstruction": []}


# Function to create a submission file.
def create_submission(out_results, data_dict):
    with open(f'submission.csv', 'w') as f:
        f.write('image_path,dataset,scene,rotation_matrix,translation_vector\n')
        for dataset in data_dict:
            if dataset in out_results:
                res = out_results[dataset]
            else:
                res = {}
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R": {}, "t": {}}
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        print(image)
                        R = scene_res[image]['R'].reshape(-1)
                        T = scene_res[image]['t'].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    f.write(
                        f'{image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n')


gc.collect()
datasets = []
for dataset in data_dict:
    datasets.append(dataset)

for dataset in datasets:
    print(dataset)
    if dataset not in out_results:
        out_results[dataset] = {}
    for scene in data_dict[dataset]:
        print(scene)
        # Fail gently if the notebook has not been submitted and the test data is not populated.
        # You may want to run this on the training data in that case?
        img_dir = f'{src}/train/{dataset}/{scene}/images'
        if not os.path.exists(img_dir):
            continue
        # Wrap the meaty part in a try-except block.
        try:
            out_results[dataset][scene] = {}
            # -----train-----
            img_fnames = [
                f'{src}/train/{x}' for x in data_dict[dataset][scene]]
            # -----test-----
            # img_fnames = [f'{src}/test/{x}' for x in data_dict[dataset][scene]]
            print(f"Got {len(img_fnames)} images")
            feature_dir = f'featureout/{dataset}_{scene}'
            if not os.path.isdir(feature_dir):
                os.makedirs(feature_dir, exist_ok=True)
            t = time()
            index_pairs = get_image_pairs_shortlist(img_fnames,
                                                    sim_th=0.5,  # should be strict
                                                    min_pairs=20,  # we select at least min_pairs PER IMAGE with biggest similarity
                                                    exhaustive_if_less=20,
                                                    device=device)
            t = time() - t
            timings['shortlisting'].append(t)
            print(f'{len(index_pairs)}, pairs to match, {t:.4f} sec')
            gc.collect()
            t = time()
            # if LOCAL_FEATURE != 'LoFTR':
            #     detect_features(img_fnames,
            #                     2048,
            #                     feature_dir=feature_dir,
            #                     upright=True,
            #                     device=device,
            #                     resize_small_edge_to=600
            #                    )
            #     gc.collect()
            #     t=time() -t
            #     timings['feature_detection'].append(t)
            #     print(f'Features detected in  {t:.4f} sec')
            #     t=time()
            #     match_features(img_fnames, index_pairs, feature_dir=feature_dir,device=device)
            # else:
            matcher(img_fnames, index_pairs, feature_dir=feature_dir,
                        device=device, resize_to_=(600, 800))
            # t=time() -t
            # timings['feature_matching'].append(t)
            # print(f'Features matched in  {t:.4f} sec')
            # database_path = f'{feature_dir}/colmap.db'
            # if os.path.isfile(database_path):
            #     os.remove(database_path)
            # gc.collect()
            # import_into_colmap(img_dir, feature_dir=feature_dir,database_path=database_path)
            # output_path = f'{feature_dir}/colmap_rec_{LOCAL_FEATURE}'

            # t=time()
            # pycolmap.match_exhaustive(database_path)
            # t=time() - t
            # timings['RANSAC'].append(t)
            # print(f'RANSAC in  {t:.4f} sec')

            # t=time()
            # # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
            # mapper_options = pycolmap.IncrementalMapperOptions()
            # mapper_options.min_model_size = 3
            # os.makedirs(output_path, exist_ok=True)
            # maps = pycolmap.incremental_mapping(database_path=database_path, image_path=img_dir, output_path=output_path, options=mapper_options)
            # print(maps)
            # #clear_output(wait=False)
            # t=time() - t
            # timings['Reconstruction'].append(t)
            # print(f'Reconstruction done in  {t:.4f} sec')
            # imgs_registered  = 0
            # best_idx = None
            # print ("Looking for the best reconstruction")
            # if isinstance(maps, dict):
            #     for idx1, rec in maps.items():
            #         print (idx1, rec.summary())
            #         if len(rec.images) > imgs_registered:
            #             imgs_registered = len(rec.images)
            #             best_idx = idx1
            # if best_idx is not None:
            #     print (maps[best_idx].summary())
            #     for k, im in maps[best_idx].images.items():
            #         key1 = f'{dataset}/{scene}/images/{im.name}'
            #         out_results[dataset][scene][key1] = {}
            #         out_results[dataset][scene][key1]["R"] = deepcopy(im.rotmat())
            #         out_results[dataset][scene][key1]["t"] = deepcopy(np.array(im.tvec))
            # print(f'Registered: {dataset} / {scene} -> {len(out_results[dataset][scene])} images')
            # print(f'Total: {dataset} / {scene} -> {len(data_dict[dataset][scene])} images')
            # create_submission(out_results, data_dict)
            # gc.collect()
        except:
            pass

# create_submission(out_results, data_dict)
