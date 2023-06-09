import dnnlib
import legacy
from torch_utils import misc
import sys
import torch

resume_pkl = sys.argv[1]
output_pkl = sys.argv[2]

with dnnlib.util.open_url(resume_pkl) as f:
    resume_data = legacy.load_network_pkl(f)

d_state_dict = resume_data['D'].state_dict()
del d_state_dict['b4.out.weight']
del d_state_dict['b4.out.bias']

snapshot_data=dict(G=resume_data['G'].state_dict(), D=d_state_dict)

torch.save(snapshot_data, output_pkl)
