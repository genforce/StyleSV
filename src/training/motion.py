from typing import Dict

import numpy as np
import torch
import random
import torch.nn as nn
from omegaconf import DictConfig

from src.torch_utils import misc
from src.torch_utils import persistence
from src.training.bspline import Bspline
from src.training.layers import (
    MappingNetwork,
    EqLRConv1d,
    FullyConnectedLayer,
)

#----------------------------------------------------------------------------

@persistence.persistent_class
class MotionMappingNetwork(torch.nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        assert self.cfg.motion.gen_strategy in ["autoregressive", "conv"], f"Unknown generation strategy: {self.cfg.motion.gen_strategy}"

        self.time_encoder=None
        if self.cfg.motion.fourier:
            self.time_encoder = AlignedTimeEncoder(
                cfg=self.cfg,
                latent_dim=self.cfg.motion.v_dim
            )
        else:
            self.mapping = MappingNetwork(
                z_dim=self.cfg.motion.z_dim,
                c_dim=self.cfg.c_dim,
                w_dim=self.cfg.motion.v_dim,
                num_ws=None,
                num_layers=2,
                activation='lrelu',
                w_avg_beta=None,
                cfg=self.cfg,
            )

        if self.cfg.motion.gen_strategy == 'autoregressive':
            self.rnn = nn.LSTM(
                input_size=self.cfg.motion.z_dim + self.cfg.c_dim,
                hidden_size=self.cfg.motion.z_dim,
                bidirectional=False,
                batch_first=True)
            self._parameters_flattened = False
            self.num_additional_codes = 0
        elif self.cfg.motion.gen_strategy == 'conv':
            # Using Conv1d without paddings instead of LSTM makes the generations good for any time in t \in (0, +\infty),
            # while LSTM would diverge for large `t`
            # Also, this allows us to use equalized learning rates
            self.conv = nn.Sequential(
                EqLRConv1d(self.cfg.motion.z_dim + self.cfg.c_dim, self.cfg.motion.z_dim, self.cfg.motion.kernel_size, padding=0, activation='lrelu', lr_multiplier=0.01),
                EqLRConv1d(self.cfg.motion.z_dim, self.cfg.motion.v_dim, self.cfg.motion.kernel_size, padding=0, activation='lrelu', lr_multiplier=0.01),
            )
            self.num_additional_codes = (self.cfg.motion.kernel_size - 1) * 2
        else:
            raise NotImplementedError(f'Unknown generation strategy: {self.cfg.motion.gen_strategy}')

    def get_max_traj_len(self, t: torch.Tensor) -> int:
        max_t = max(self.cfg.sampling.max_num_frames - 1, t.max().item()) # [1]
        max_traj_len = np.ceil(max_t / self.cfg.motion.motion_z_distance).astype(int).item() + 2 # [1]
        return max_traj_len

    def generate_motion_u_codes(self, c: torch.Tensor, t: torch.Tensor, motion_z: torch.Tensor=None) -> Dict:
        """
        Arguments:
            - c of shape [batch_size, c_dim]
            - t of shape [batch_size, num_frames]
            - w of shape [batch_size, w_dim]
            - motion_z of shape [batch_size, max_traj_len, motion_z_dim] --- in case we want to reuse some existing motion noise
        """
        out = {}
        batch_size, num_frames = t.shape

        # Consutruct trajectories (from code idx for now)
        max_traj_len = self.get_max_traj_len(t) + self.num_additional_codes # [1]

        if motion_z is None:
            motion_z = torch.randn(batch_size, max_traj_len, self.cfg.motion.z_dim, device=c.device) # [batch_size, max_traj_len, motion.z_dim]

        # Input motion trajectories are just random noise
        input_trajs = motion_z[:batch_size, :max_traj_len, :self.cfg.motion.z_dim].to(c.device) # [batch_size, max_traj_len, motion.z_dim]

        if self.cfg.c_dim > 0:
            # Different classes might have different motions, so it should be useful to condition on c
            misc.assert_shape(c, [batch_size, None])
            input_trajs = torch.cat([input_trajs, c.unsqueeze(1).repeat(1, max_traj_len, 1)], dim=2) # [batch_size, max_traj_len, motion.z_dim + cond_dim]

        if self.cfg.motion.gen_strategy == 'autoregressive':
            # Somehow, RNN parameters do not get flattened on their own and we get a lot of warnings...
            if not self._parameters_flattened:
                self.rnn.flatten_parameters()
                self._parameters_flattened = True
            trajs, _ = self.rnn(input_trajs) # [batch_size, max_traj_len, motion.z_dim]
        elif self.cfg.motion.gen_strategy == 'conv':
            trajs = self.conv(input_trajs.permute(0, 2, 1)).permute(0, 2, 1) # [batch_size, max_traj_len, motion.v_dim]
        else:
            raise NotImplementedError(f'Unknown generation strategy: {self.cfg.motion.gen_strategy}')

        # Now, we should select neighbouring codes for each frame
        left_idx = (t / self.cfg.motion.motion_z_distance).floor().long() # [batch_size, num_frames]
        batch_idx = torch.arange(batch_size, device=c.device).unsqueeze(1).repeat(1, num_frames) # [batch_size, num_frames]
        motion_u_left = trajs[batch_idx, left_idx] # [batch_size, num_frames, motion.z_dim]
        motion_u_right = trajs[batch_idx, left_idx + 1] # [batch_size, num_frames, motion.z_dim]

        # Compute `u` codes as the interpolation between `u_left` and `u_right`
        t_left = t - t % self.cfg.motion.motion_z_distance # [batch_size, num_frames]
        t_right = t_left + self.cfg.motion.motion_z_distance # [batch_size, num_frames]
        # Compute interpolation weights `alpha` (we'll use them later)
        interp_weights = ((t % self.cfg.motion.motion_z_distance) / self.cfg.motion.motion_z_distance).unsqueeze(2).to(torch.float32) # [batch_size, num_frames, 1]
        motion_u = motion_u_left * (1 - interp_weights) + motion_u_right * interp_weights # [batch_size, num_frames, motion.z_dim]
        motion_u = motion_u.view(batch_size * num_frames, motion_u.shape[2]).to(torch.float32) # [batch_size * num_frames, motion.z_dim]

        # Save the results into the output dict
        out['motion_u_left'] = motion_u_left # [batch_size, num_frames, motion.z_dim]
        out['motion_u_right'] = motion_u_right # [batch_size, num_frames, motion.z_dim]
        out['t_left'] = t_left # [batch_size, num_frames]
        out['t_right'] = t_right # [batch_size, num_frames]
        out['interp_weights'] = interp_weights # [batch_size, num_frames, 1]
        out['motion_u'] = motion_u # [batch_size * num_frames, motion.z_dim]
        out['motion_z'] = motion_z # [batch_size+, max_traj_len+, motion.z_dim+]

        return out

    def get_dim(self) -> int:
        return self.cfg.motion.v_dim if self.time_encoder is None else self.time_encoder.get_dim()

    def forward(self, c: torch.Tensor, t: torch.Tensor, motion_z: Dict=None) -> Dict:
        assert len(c) == len(t), f"Wrong shape: {c.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        out = {} # We'll be aggregating the result here
        motion_u_info: Dict = self.generate_motion_u_codes(c, t, motion_z=motion_z) # Dict of tensors
        motion_u = motion_u_info['motion_u'].view(t.shape[0] * t.shape[1], -1) # [batch_size * num_frames, motion.z_dim]

        # Compute the `v` motion codes
        if self.cfg.motion.fourier:
            motion_v = self.time_encoder(
                t=t,
                motion_u_left=motion_u_info['motion_u_left'],
                motion_u_right=motion_u_info['motion_u_right'],
                t_left=motion_u_info['t_left'],
                t_right=motion_u_info['t_right'],
                interp_weights=motion_u_info['interp_weights'],
            ) # [batch_size * num_frames, motion_v_dim]
        else:
            motion_v = self.mapping(z=motion_u, c=c.repeat_interleave(t.shape[1], dim=0)) # [batch_size * num_frames, motion.v_dim]

        out['motion_v'] = motion_v # [batch_size * num_frames, motion.v_dim]
        out['motion_z'] = motion_u_info['motion_z'] # (Any shape)

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class AlignedTimeEncoder(nn.Module):
    def __init__(self,
        latent_dim: int=512,
        cfg: DictConfig = {},
    ):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim

        freqs = construct_linspaced_frequencies(self.cfg.time_enc.dim, self.cfg.time_enc.min_period_len, self.cfg.time_enc.max_period_len)
        self.register_buffer('freqs', freqs) # [1, num_fourier_feats]

        # Creating the affine without bias to prevent motion mode collapse
        self.periods_predictor = FullyConnectedLayer(latent_dim, freqs.shape[1], activation='linear', bias=False)
        self.phase_predictor = FullyConnectedLayer(latent_dim, freqs.shape[1], activation='linear', bias=False)
        period_lens = 2 * np.pi / self.freqs # [1, num_fourier_feats]
        phase_scales = self.cfg.time_enc.max_period_len / period_lens # [1, num_fourier_feats]
        self.register_buffer('phase_scales', phase_scales)

        self.aligners_predictor = FullyConnectedLayer(latent_dim, self.freqs.shape[1] * 2, activation='linear', bias=False)

    def get_dim(self) -> int:
        return self.freqs.shape[1] * 2

    def forward(self, t: torch.Tensor, motion_u_left: torch.Tensor, motion_u_right: torch.Tensor, interp_weights: torch.Tensor, t_left: torch.Tensor, t_right: torch.Tensor):
        batch_size, num_frames, motion_u_dim = motion_u_left.shape # [1], [1], [1]

        misc.assert_shape(t, [batch_size, num_frames])
        misc.assert_shape(motion_u_left, [batch_size, num_frames, None])
        misc.assert_shape(motion_u_right, [batch_size, num_frames, None])
        misc.assert_shape(interp_weights, [batch_size, num_frames, 1])
        assert t.shape == t_left.shape == t_right.shape, f"Wrong shape: {t.shape} vs {t_left.shape} vs {t_right.shape}"

        motion_u_left = motion_u_left.view(batch_size * num_frames, motion_u_dim) # [batch_size * num_frames, motion_u_dim]
        motion_u_right = motion_u_right.view(batch_size * num_frames, motion_u_dim) # [batch_size * num_frames, motion_u_dim]
        periods = self.periods_predictor(motion_u_left).tanh() + 1 # [batch_size * num_frames, feat_dim]
        phases = self.phase_predictor(motion_u_left) # [batch_size * num_frames, feat_dim]
        aligners_left = self.aligners_predictor(motion_u_left) # [batch_size * num_frames, out_dim]
        aligners_right = self.aligners_predictor(motion_u_right) # [batch_size * num_frames, out_dim]

        raw_pos_embs = self.freqs * periods * t.view(-1).float().unsqueeze(1) + phases * self.phase_scales # [bf, feat_dim]
        raw_pos_embs_left = self.freqs * periods * t_left.view(-1).float().unsqueeze(1) + phases * self.phase_scales # [bf, feat_dim]
        raw_pos_embs_right = self.freqs * periods * t_right.view(-1).float().unsqueeze(1) + phases * self.phase_scales # [bf, feat_dim]

        pos_embs = torch.cat([raw_pos_embs.sin(), raw_pos_embs.cos()], dim=1) # [bf, out_dim]
        pos_embs_left = torch.cat([raw_pos_embs_left.sin(), raw_pos_embs_left.cos()], dim=1) # [bf, out_dim]
        pos_embs_right = torch.cat([raw_pos_embs_right.sin(), raw_pos_embs_right.cos()], dim=1) # [bf, out_dim]

        interp_weights = interp_weights.view(-1, 1) # [bf, 1]
        aligners_remove = pos_embs_left * (1 - interp_weights) + pos_embs_right * interp_weights # [bf, out_dim]
        aligners_add = aligners_left * (1 - interp_weights) + aligners_right * interp_weights # [bf, out_dim]
        time_embs = pos_embs - aligners_remove + aligners_add # [bf, out_dim]

        return time_embs

#----------------------------------------------------------------------------

def construct_linspaced_frequencies(num_freqs: int, min_period_len: int, max_period_len: int) -> torch.Tensor:
    freqs = 2 * np.pi / (2 ** np.linspace(np.log2(min_period_len), np.log2(max_period_len), num_freqs)) # [num_freqs]
    freqs = torch.from_numpy(freqs[::-1].copy().astype(np.float32)).unsqueeze(0) # [1, num_freqs]

    return freqs

#----------------------------------------------------------------------------

@persistence.persistent_class
class BSplineMotionMappingNetwork(MotionMappingNetwork):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if self.cfg.motion.fourier:
            self.time_encoder = BSplineAlignedTimeEncoder(cfg=self.cfg, latent_dim=self.cfg.motion.v_dim)
        self.B = Bspline([0,0,0,1,2,3,4,5,6,7,7,7], 2)

    def get_max_traj_len(self, t: torch.Tensor) -> int:
        # for b spline, we create dump head and tail additionally
        ret = super().get_max_traj_len(t)
        return ret + 2

    def generate_motion_u_codes(self, c: torch.Tensor, t: torch.Tensor, motion_z: torch.Tensor=None) -> Dict:
        """
        Arguments:
            - c of shape [batch_size, c_dim]
            - t of shape [batch_size, num_frames]
            - w of shape [batch_size, w_dim]
            - motion_z of shape [batch_size, max_traj_len, motion_z_dim] --- in case we want to reuse some existing motion noise
        """
        out = {}
        batch_size, num_frames = t.shape

        # Consutruct trajectories (from code idx for now)
        max_traj_len = self.get_max_traj_len(t) + self.num_additional_codes # [1]

        if motion_z is None:
            motion_z = torch.randn(batch_size, max_traj_len, self.cfg.motion.z_dim, device=c.device) # [batch_size, max_traj_len, motion.z_dim]

        # Input motion trajectories are just random noise
        input_trajs = motion_z[:batch_size, :max_traj_len, :self.cfg.motion.z_dim].to(c.device) # [batch_size, max_traj_len, motion.z_dim]

        if self.cfg.c_dim > 0:
            # Different classes might have different motions, so it should be useful to condition on c
            misc.assert_shape(c, [batch_size, None])
            input_trajs = torch.cat([input_trajs, c.unsqueeze(1).repeat(1, max_traj_len, 1)], dim=2) # [batch_size, max_traj_len, motion.z_dim + cond_dim]

        if self.cfg.motion.gen_strategy == 'autoregressive':
            # Somehow, RNN parameters do not get flattened on their own and we get a lot of warnings...
            if not self._parameters_flattened:
                self.rnn.flatten_parameters()
                self._parameters_flattened = True
            trajs, _ = self.rnn(input_trajs) # [batch_size, max_traj_len, motion.z_dim]
        elif self.cfg.motion.gen_strategy == 'conv':
            trajs = self.conv(input_trajs.permute(0, 2, 1)).permute(0, 2, 1) # [batch_size, max_traj_len, motion.v_dim]
        else:
            raise NotImplementedError(f'Unknown generation strategy: {self.cfg.motion.gen_strategy}')

        # Now, we should select neighbouring codes for each frame
        # b spline: [start_idx, start_idx+1, start_idx+2] @ B.collmat(time)
        start_idx = (t / self.cfg.motion.motion_z_distance).floor().long() # [batch_size, num_frames]
        batch_idx = torch.arange(batch_size, device=c.device).unsqueeze(1).repeat(1, num_frames) # [batch_size, num_frames]
        motion_u0 = trajs[batch_idx, start_idx] # [batch_size, num_frames, motion.z_dim]
        motion_u1 = trajs[batch_idx, start_idx + 1] # [batch_size, num_frames, motion.z_dim]
        motion_u2 = trajs[batch_idx, start_idx + 2] # [batch_size, num_frames, motion.z_dim]

        # Compute `u` codes as the interpolation between `u_left` and `u_right`
        t0 = t - t % self.cfg.motion.motion_z_distance # [batch_size, num_frames]
        t1 = t0 + self.cfg.motion.motion_z_distance # [batch_size, num_frames]
        t2 = t0 + 2 * self.cfg.motion.motion_z_distance # [batch_size, num_frames]
        # Compute interpolation weights `alpha` (we'll use them later)
        pct = (t % self.cfg.motion.motion_z_distance) / self.cfg.motion.motion_z_distance
        lpct = pct.unsqueeze(2).to(torch.float32) 
        pct = pct.view(-1) + 2
        interp_weights = self.B.collmat(pct.cpu().numpy())[:, 2: 5]
        w0, w1, w2 = torch.split(torch.tensor(interp_weights).to(pct.device).view(batch_size, num_frames, 3), 1, dim=-1)   # [batch_size, num_frames, 1]
        
        motion_u = motion_u0 * w0 + motion_u1 * w1 + motion_u2 * w2
        motion_u = motion_u.view(batch_size * num_frames, motion_u.shape[2]).to(torch.float32) # [batch_size * num_frames, motion.z_dim]

        # Save the results into the output dict
        out['motion_u0'] = motion_u0 # [batch_size, num_frames, motion.z_dim]
        out['motion_u1'] = motion_u1 # [batch_size, num_frames, motion.z_dim]
        out['motion_u2'] = motion_u2 # [batch_size, num_frames, motion.z_dim]
        out['t0'] = t0 # [batch_size, num_frames]
        out['t1'] = t1 # [batch_size, num_frames]
        out['t2'] = t2 # [batch_size, num_frames]
        out['w0'] = w0 # [batch_size, num_frames]
        out['w1'] = w1 # [batch_size, num_frames]
        out['w2'] = w2 # [batch_size, num_frames]
        out['weights'] = lpct
        out['motion_u'] = motion_u # [batch_size * num_frames, motion.z_dim]
        out['motion_z'] = motion_z # [batch_size+, max_traj_len+, motion.z_dim+]

        return out

    def forward(self, c: torch.Tensor, t: torch.Tensor, motion_z: Dict=None) -> Dict:
        assert len(c) == len(t), f"Wrong shape: {c.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        out = {} # We'll be aggregating the result here
        motion_u_info: Dict = self.generate_motion_u_codes(c, t, motion_z=motion_z) # Dict of tensors
        motion_u = motion_u_info['motion_u'].view(t.shape[0] * t.shape[1], -1) # [batch_size * num_frames, motion.z_dim]

        # Compute the `v` motion codes
        if self.cfg.motion.fourier:
            motion_v = self.time_encoder(
                t=t,
                motion_u0=motion_u_info['motion_u0'],
                motion_u1=motion_u_info['motion_u1'],
                motion_u2=motion_u_info['motion_u2'],
                t0=motion_u_info['t0'],
                t1=motion_u_info['t1'],
                t2=motion_u_info['t2'],
                w0=motion_u_info['w0'],
                w1=motion_u_info['w1'],
                w2=motion_u_info['w2'],
                weights=motion_u_info['weights'],
            ) # [batch_size * num_frames, motion_v_dim]
        else:
            motion_v = self.mapping(z=motion_u, c=c.repeat_interleave(t.shape[1], dim=0)) # [batch_size * num_frames, motion.v_dim]

        out['motion_v'] = motion_v # [batch_size * num_frames, motion.v_dim]
        out['motion_z'] = motion_u_info['motion_z'] # (Any shape)

        return out

@persistence.persistent_class
class BSplineAlignedTimeEncoder(AlignedTimeEncoder):
    def __init__(self,
        latent_dim: int=512,
        cfg: DictConfig = {},
    ):
        super().__init__(latent_dim=latent_dim, cfg=cfg)
        self.cfg = cfg
        self.latent_dim = latent_dim

        freqs = construct_linspaced_frequencies(self.cfg.time_enc.dim, self.cfg.time_enc.min_period_len, self.cfg.time_enc.max_period_len)
        self.register_buffer('freqs', freqs) # [1, num_fourier_feats]

        # Creating the affine without bias to prevent motion mode collapse
        self.periods_predictor = FullyConnectedLayer(latent_dim, freqs.shape[1], activation='linear', bias=False)
        self.phase_predictor = FullyConnectedLayer(latent_dim, freqs.shape[1], activation='linear', bias=False)
        period_lens = 2 * np.pi / self.freqs # [1, num_fourier_feats]
        phase_scales = self.cfg.time_enc.max_period_len / period_lens # [1, num_fourier_feats]
        self.register_buffer('phase_scales', phase_scales)

        self.aligners_predictor = FullyConnectedLayer(latent_dim, self.freqs.shape[1] * 2, activation='linear', bias=False)

        try:
            self.offset = self.cfg.offset
        except:
            self.offset = 0

    def get_dim(self) -> int:
        return self.freqs.shape[1] * 2

    def forward(self, t: torch.Tensor, motion_u0: torch.Tensor, motion_u1: torch.Tensor, motion_u2: torch.Tensor, 
                                       w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, 
                                       t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, weights: torch.Tensor,):
        batch_size, num_frames, motion_u_dim = motion_u1.shape # [1], [1], [1]

        misc.assert_shape(t, [batch_size, num_frames])
        misc.assert_shape(motion_u0, [batch_size, num_frames, None])
        misc.assert_shape(w0, [batch_size, num_frames, 1])
        assert t.shape == t0.shape == t1.shape, f"Wrong shape: {t.shape} vs {t0.shape} vs {t1.shape}"
        w0 = w0.view(-1, 1)
        w1 = w1.view(-1, 1)
        w2 = w2.view(-1, 1)
        weights = weights.view(-1,1)
        
        offset = torch.randint(0, self.offset+1, (t.shape[0], 1)).to(t.device)
        t = t + offset
        t0 = t0 + offset
        t1 = t1 + offset
        t2 = t2 + offset

        motion_u0 = motion_u0.view(batch_size * num_frames, motion_u_dim) # [batch_size * num_frames, motion_u_dim]
        motion_u1 = motion_u1.view(batch_size * num_frames, motion_u_dim) # [batch_size * num_frames, motion_u_dim]
        motion_u2 = motion_u2.view(batch_size * num_frames, motion_u_dim) # [batch_size * num_frames, motion_u_dim]
        periods0 = self.periods_predictor(motion_u0).tanh() + 1 # [batch_size * num_frames, feat_dim]
        phases0 = self.phase_predictor(motion_u0) # [batch_size * num_frames, feat_dim]

        aligners0 = self.aligners_predictor(motion_u0) # [batch_size * num_frames, out_dim]
        aligners1 = self.aligners_predictor(motion_u1) # [batch_size * num_frames, out_dim]
        aligners2 = self.aligners_predictor(motion_u2) # [batch_size * num_frames, out_dim]
        aligners_add = aligners0 * w0 + aligners1 * w1 + aligners2 * w2
        base_f = 0

        if self.cfg.bs_emb:
            periods1 = self.periods_predictor(motion_u1).tanh() + 1 # [batch_size * num_frames, feat_dim]
            periods2 = self.periods_predictor(motion_u2).tanh() + 1 # [batch_size * num_frames, feat_dim]
            phases1 = self.phase_predictor(motion_u1) # [batch_size * num_frames, feat_dim]
            phases2 = self.phase_predictor(motion_u2) # [batch_size * num_frames, feat_dim]
            periods = periods0 * w0 + periods1 * w1 + periods2 * w2
            phases = phases0 * w0 + phases1 * w1 + phases2 * w2
            raw_pos_embs = ( base_f + self.freqs * periods) * t.view(-1).float().unsqueeze(1) + phases * self.phase_scales # [bf, feat_dim]
            pos_embs = torch.cat([raw_pos_embs.sin(), raw_pos_embs.cos()], dim=1) # [bf, out_dim]

            raw_pos_embs_left = (base_f + self.freqs * (periods0 + periods1) / 2) * t0.view(-1).float().unsqueeze(1) + (phases0 + phases1) / 2 * self.phase_scales
            raw_pos_embs_right = (base_f + self.freqs * (periods1 + periods2) / 2) * t1.view(-1).float().unsqueeze(1) + (phases1 + phases2) / 2 * self.phase_scales
            pos_embs_left = torch.cat([raw_pos_embs_left.sin(), raw_pos_embs_left.cos()], dim=1)
            pos_embs_right = torch.cat([raw_pos_embs_right.sin(), raw_pos_embs_right.cos()], dim=1)
            time_embs = pos_embs + aligners_add # [bf, out_dim]
        else:
            periods = periods0
            phases = phases0
            raw_pos_embs = self.freqs * periods * t.view(-1).float().unsqueeze(1) + phases * self.phase_scales # [bf, feat_dim]
            pos_embs = torch.cat([raw_pos_embs.sin(), raw_pos_embs.cos()], dim=1) # [bf, out_dim]
            raw_pos_embs_left = self.freqs * periods * t0.view(-1).float().unsqueeze(1) + phases * self.phase_scales # [bf, feat_dim]
            pos_embs_left = torch.cat([raw_pos_embs_left.sin(), raw_pos_embs_left.cos()], dim=1) # [bf, out_dim]
            raw_pos_embs_right = self.freqs * periods * t1.view(-1).float().unsqueeze(1) + phases * self.phase_scales # [bf, feat_dim]
            pos_embs_right = torch.cat([raw_pos_embs_right.sin(), raw_pos_embs_right.cos()], dim=1) # [bf, out_dim]
            aligners_remove = pos_embs_left * (1-weights) + pos_embs_right * weights
            time_embs = pos_embs - aligners_remove + aligners_add


        return time_embs