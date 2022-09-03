import os
import argparse
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import lmdb
from tqdm import tqdm
from imageio import imread
import shutil
import pickle
import sys
import random

sys.path.insert(0, './')

from denoiser.config import Config
from denoiser.modeling.architectures import build_architecture

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from collections import defaultdict
import torch
import faiss
from tools.nearest_search import search_raw_array_pytorch

# resource object, can be re-used over calls
res = faiss.StandardGpuResources()
# put on same stream as pytorch to avoid synchronizing streams
res.setDefaultNullStreamAllDevices()

size_stats = defaultdict(int)
format_stats = defaultdict(int)




def filter_image_sizes(images):
    filtered = []
    for idx, fname in enumerate(images):
        if (idx % 100) == 0:
            print ('loading images', idx, '/', len(images))
        try:
            with PIL.Image.open(fname) as img:
                w = img.size[0]
                h = img.size[1]
                if (w > 512 or h > 512) or (w < 256 or h < 256):
                    continue
                filtered.append((fname, w, h))
        except:
            print ('Could not load image', fname, 'skipping file..')
    return filtered


def nofilter_image_sizes(images):
    filtered = []
    for idx, fname in enumerate(images):
        if (idx % 100) == 0:
            print ('loading images', idx, '/', len(images))
        try:
            with PIL.Image.open(fname) as img:
                w = img.size[0]
                h = img.size[1]
                if (w > 512 or h > 512) or (w < 256 or h < 256):
                    continue
                filtered.append((fname, w, h))
        except:
            print ('Could not load image', fname, 'skipping file..')
    return filtered


def noisify(img,std,noise_type="gaussian",lam=30):
    if noise_type == "gaussian":
        img_noise = img + std * np.random.normal(loc=0.0, scale=1.0, size=img.shape)
    elif noise_type == "poisson":
        img_noise = np.random.poisson(img*lam) / float(lam)
    else:
        raise TypeError
    return img_noise


def shift_concat_image(img_noise, patch_size):
    if len(img_noise.shape) < 3:
        [H, W] = img_noise.shape
        C = 1
        img_noise = img_noise.reshape([H, W, C])
    else:
        [H, W, C] = img_noise.shape
    img_noise_pad = np.pad(img_noise, pad_width=((patch_size, patch_size),
                                                (patch_size, patch_size),
                                                (0, 0)),
                           mode="reflect")
    patches = np.zeros([H, W, patch_size * patch_size, C])
    for i in range(-(patch_size - 1) // 2, (patch_size - 1) // 2 + 1):
        for j in range(-(patch_size - 1) // 2, (patch_size - 1) // 2 + 1):
            if i == 0 and j == 0:
                continue  # exclude the center pixel
            h_start = max(0, i + patch_size)
            h_end = min(H + 2 * patch_size, i + patch_size + H)
            w_start = max(0, j + patch_size)
            w_end = min(W + 2 * patch_size, j + patch_size + W)

            pi = i + (patch_size - 1) // 2
            pj = j + (patch_size - 1) // 2

            patches[:, :, (pi * patch_size + pj), :] = \
                img_noise_pad[h_start:h_end, w_start:w_end, :]

    return patches


def search_nlm_images_gpu(img, patches, num_select):
    [H, W, D, C] = patches.shape
    patches = patches.reshape([H * W, D * C]).astype(np.float32)
    patches = torch.from_numpy(patches).to("cuda:0")
    dist, ind_y = search_raw_array_pytorch(res, patches, patches, num_select, metric=faiss.METRIC_L2)

    images_sim = np.zeros([num_select, H * W, C]).astype(np.float32)

    for s in range(num_select):
        images_sim[s, :, :] = img.reshape([H * W, C])[ind_y[:, s].cpu().numpy(), :]

    images_sim = images_sim.reshape([num_select, H, W, C])
    return images_sim

def compute_sim_images(img, patch_size, num_select, img_ori=None):
    if img_ori is not None:
        patches = shift_concat_image(img_ori, patch_size)
    else:
        patches = shift_concat_image(img, patch_size)
    images_sim = search_nlm_images_gpu(img, patches, num_select+1)
    return images_sim[1::, ...]


def clip_to_unit8(arr):
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def read_data_file(data_file):
    f = open(data_file, 'r')
    lines = f.readlines()
    data_pathes = []
    for line in lines:
        data_pathes.append(line[0:-1])
    return data_pathes


def _get_keys_shapes_pickle(meta_info_file):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(meta_info_file, 'rb'))
    keys = meta_info['keys']
    shapes = meta_info['shapes']
    return keys, shapes


def _read_img_noise_lmdb(env, key, shape, dtype=np.uint8):
    with env.begin(write=False) as txn:
        buf = txn.get("{}_noise".format(key).encode('ascii'))
    img_noise_flat = np.frombuffer(buf, dtype=dtype)
    H, W, C = shape
    img_noise = img_noise_flat.reshape(H, W, C)
    return img_noise

def subsample(i_index,j_index,img):
    for i in range(i_index):
        for j in range(j_index):
            if((i+j)%2==0):
                img[i,j] = 0
    return img
def subsample2(i_index,j_index,img):
    for i in range(i_index):
        for j in range(j_index):
            if random.randint(1,2)==1:
                img[i,j]=0
    return img

def subsample1(i_index,j_index,img,img2,img3):
    for i in range(i_index):
        for j in range(j_index):
            if random.randint(1,2)==1:
                img[i,j]=0
                img2[i,j]=0
                img3[i,j]=0

    return np.concatenate((img.reshape(1,128,128),img2.reshape(1,128,128),img3.reshape(1,128,128)),axis=0)

def main():
    parser = argparse.ArgumentParser(
        description='Compute the similarity images and save them into a LMDB dataset file.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-folder", default='./datasets/Train')
    parser.add_argument("--output-folder", default='./datasets')
    parser.add_argument("--noise-type", default="gaussian")
    parser.add_argument("--std", default=1.0)
    parser.add_argument("--lam", default=30)
    parser.add_argument("--num-sim", default=6)
    parser.add_argument("--patch-size", default=5)
    parser.add_argument("--max-files", default=1000)
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--model_weight", default=None)
    parser.add_argument("--lmdb_file", default=None)
    parser.add_argument("--key_file", default=None)
    parser.add_argument("--data_num", default=3600)
    args = parser.parse_args()

#     img_files = os.listdir(args.data_folder)
    num_images = args.data_num

    if args.noise_type == "gaussian":
        noise_param = args.std
    elif args.noie_type == "poisson":
        noise_param = args.lam
    else:
        raise TypeError

    args.output_file = "{}/seg_0610gus_{}{}_ps{}_ns{}_lmdb".format(args.output_folder, args.noise_type,
                                                                  noise_param, args.patch_size, args.num_sim)

    if args.config_file is None:
        
        data_in_test = np.load("args.data_folder")
        img = data_in_test.reshape(args.data_num,128,128,1)
        data_noise = noisify(img, std = 0.6, noise_type=args.noise_type, lam=args.lam)
   

    data_noise = np.array(data_noise).astype(np.float32)
#     data_noise_norm = (data_noise - data_noise.min()) / (data_noise.max() - data_noise.min())
    data_noise_norm = data_noise
    # ----------------------------------------------------------
    if os.path.exists(args.output_file):
        print("{} exists, deleted...".format(args.output_file))
        shutil.rmtree(args.output_file)

    commit_interval = 10

    # Estimate the lmdb size.
    data_nbytes = data_noise.astype(np.float32).nbytes
    data_size = data_nbytes * (args.num_sim + 1)

    env = lmdb.open(args.output_file, map_size=data_size*1.5)

    txn = env.begin(write=True)
    shapes = []
    tqdm_iter = tqdm(enumerate(range(num_images)), total=args.data_num, leave=False)

    keys = []
    for idx, key in tqdm_iter:

        tqdm_iter.set_description('Write {}'.format(key))
        keys.append(str(key))

        img_noise = data_noise[idx]
        img_noise_norm = data_noise_norm[idx]

        if args.config_file is not None:
            img_noise_norm_gpu = torch.from_numpy(img_noise_norm).to(torch.float32).to(device).reshape(1, 1, img_noise_norm.shape[0], img_noise_norm.shape[1])
            with torch.no_grad():
                img_noise_norm = model(img_noise_norm_gpu)
                img_noise_norm = img_noise_norm.cpu().numpy().squeeze()

#         img_noise_sim33 = compute_sim_images(img_noise, patch_size=3, num_select=args.num_sim,
#                                            img_ori=img_noise_norm)
        
#         img_noise_sim = compute_sim_images(img_noise, patch_size=5, num_select=args.num_sim)
        
        img_noise_sim_1 = compute_sim_images(img_noise, patch_size=3, num_select=args.num_sim)
        img_noise_sim_2 = compute_sim_images(img_noise, patch_size=5, num_select=args.num_sim)
        img_noise_sim_3 = compute_sim_images(img_noise, patch_size=7, num_select=args.num_sim)

        img_noise_sim1_1 = img_noise_sim_1[random.randint(0,args.num_sim-1),:,:,:].reshape(128,128)
        img_noise_sim2_1 = img_noise_sim_2[random.randint(0,args.num_sim-1),:,:,:].reshape(128,128)
        img_noise_sim3_1 = img_noise_sim_3[random.randint(0,args.num_sim-1),:,:,:].reshape(128,128)
        
#         img_noise_sim1_2 = img_noise_sim_1[random.randint(0,args.num_sim-1),:,:,:].reshape(128,128)
#         img_noise_sim2_2 = img_noise_sim_2[random.randint(0,args.num_sim-1),:,:,:].reshape(128,128)
#         img_noise_sim3_2 = img_noise_sim_3[random.randint(0,args.num_sim-1),:,:,:].reshape(128,128)
        
        randidx = random.randint(1,3)
        if randidx==1:
            img_noise_sim1 = img_noise_sim1_1.copy()
            img_noise_sim2 = img_noise_sim1_1.copy()
        elif randidx==2:
            img_noise_sim1 = img_noise_sim2_1.copy()
            img_noise_sim2 = img_noise_sim2_1.copy()
        else:
            img_noise_sim1 = img_noise_sim3_1.copy()
            img_noise_sim2 = img_noise_sim3_1.copy()
            
        sim1_temp1 = img_noise_sim1.copy()
        sim1_temp2 = img_noise_sim2.copy()
        sim1_temp2_copy = img_noise_sim2.copy()

        
        img_noise1 = img_noise.reshape(128,128)
        noise1_temp1 = img_noise1.copy()
        noise1_temp2 = img_noise1.copy()
        
        con1 = subsample1(128,128,sim1_temp1,sim1_temp2,noise1_temp1)
        
        noise1 = con1[2].reshape(128,128)
        noise2 = noise1_temp2 - noise1
        
        sim1 = con1[0].reshape(128,128)   
        sim2 = sim1_temp2_copy-con1[1].reshape(128,128)
        
        pair1 = noise1 + sim2
        pair2 = noise2 + sim1

        pair1 = pair1.reshape(1,128,128,1)
        pair2 = pair2.reshape(128,128,1)
        print(img_noise_sim1.shape,img_noise.shape)

        
        
        key_noise_byte = "{}_noise".format(key).encode('ascii')
        key_noise_sim_byte = "{}_noise_sim".format(key).encode('ascii')

        txn.put(key_noise_byte, pair2)
        txn.put(key_noise_sim_byte, pair1)

        H, W, C= img_noise.shape
        shapes.append('{:d}_{:d}_{:d}'.format(H, W, C))


        if (idx + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish writing lmdb')

    meta_info = {"shapes": shapes,
                 "keys": keys,
                 "num_sim": args.num_sim,
                 "patch_size": args.patch_size}

    pickle.dump(meta_info, open("{}_seg_0610gus_meta_info.pkl".format(args.output_file), "wb"))
    print('Finish creating lmdb meta info.')


if __name__ == "__main__":
    main()





