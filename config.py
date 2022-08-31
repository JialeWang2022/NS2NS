target_type = "noise-similarity"
model_name = "NS2NS_denoiser".format(target_type)
model_weight = None
workers = 4
epochs = 1000
start_epoch = 0
batch_size = 64
crop_size = 128
num_channel = 1
num_sim = 1
num_select = 1
print_freq = 10
test_freq = 5
resume = None
world_size = 1
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = None
multiprocessing_distributed = True


data_train = dict(
    type="lmdb",
    lmdb_file="./datasets/seg3600_0610gus_ran35_gaussian1.0_ps5_ns8_lmdb",
    meta_info_file="./datasets/seg3600_0610gus_ran35_gaussian1.0_ps5_ns8_lmdb_seg3600_0610gus_ran35_meta_info.pkl",
    crop_size=crop_size,
    target_type=target_type,
    random_flip=False,
    prune_dataset=None,
    num_sim=num_sim,
    num_select=num_select,
    load_data_all=False,
    incorporate_noise=False,
    dtype="float32",
    ims_per_batch=batch_size,
    shuffle=True,
    train=True,
)

data_test = dict(
    type="bsd_npy",
    data_file='./datasets/test/noisy_0610_37.npy',
    target_file='./datasets/test/clean_0610_37.npy',
#     norm=[0.0, 1.0],
    shuffle=False,
    ims_per_batch=37,
    train=False,
)

model = dict(
    type="common_denoiser",
    base_net=dict(
        type="unet",
        n_channels=1,
        n_classes=1,
        activation_type="leaky_relu",
        bilinear=False,
        residual=True,
        use_bn=True
    ),

    denoiser_head=dict(
        head_type="supervise",
        loss_type="l1",
        loss_weight={"l1": 1},
    ),

    weight=None,
)

solver = dict(
    type="adam",
    base_lr=0.0001,
    bias_lr_factor=1,
    betas=(0.1, 0.99),
    weight_decay=0,
    weight_decay_bias=0,
    lr_type="ramp",
    max_iter=epochs,
    ramp_up_fraction=0.1,
    ramp_down_fraction=0.3,
)

results = dict(
    output_dir="./results/{}".format(model_name),
)