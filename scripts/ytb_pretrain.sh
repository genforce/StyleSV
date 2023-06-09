python src/infra/launch.py \
hydra.run.dir=. \
exp_suffix=ytb256_stylesv_pretrain_05_bgc \
env=local slurm=false \
training.mirror=false \
training.batch_size=64 \
training.num_workers=1 \
dataset=ytb \
dataset.path=${YOURPATH}/data/ytb_256.zip \
dataset.resolution=256 \
sampling.num_frames_per_video=1 \
model.generator.time_enc.min_period_len=256 model.generator.freezesyn=0 \
model.generator.bspline=false model.generator.bs_emb=false \
model.generator.fmaps=0.5 model.discriminator.fmaps=0.5 model.optim.generator.lr=0.0025 \
model.generator.learnable_motion_mask=false model.generator.init_motion_mask=zeros model.generator.fuse_w=concat \
model.discriminator.tsm=false \
model.discriminator.tmean=true \
model.loss_kwargs.r1_gamma=0.5 \
num_gpus=8 training.snap=100 \
training.augpipe=bgc \
project_release_dir=${YOURPATH}/videogen/work_dirs/ytb256_stylesv_pretrain_05_bgc
