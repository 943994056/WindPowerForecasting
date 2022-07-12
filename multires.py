import torch
from torch import nn
from Dlinear import Model as Dlinear

class Multi_Res(nn.Module):
    def __init__(self, cfg, downsample_list=[1,2,4,8,14]) -> None:
        super(Multi_Res, self).__init__() # input 144*7, downsample 144*7->144
        self.base_input_len = cfg['input_len']
        self.pred_len = cfg['output_len']
        self.downsample_list = downsample_list
        self.channels = cfg['in_var']
        for downsample in downsample_list:
            self.add_module(f'downsample_{downsample}', nn.AvgPool1d(downsample))
            self.add_module(f'Dlinear_{downsample}', Dlinear(int(self.base_input_len/downsample), self.pred_len, self.channels, individual=False))
    def forward(self, x):
        y = torch.zeros_like(x)
        for downsample in self.downsample_list:
            xx = x[:, -self.base_input_len*downsample:, :]
            xx = xx.permute(0,2,1)
            xx = self._modules[f'downsample_{downsample}'](xx)
            xx = xx.permute(0,2,1)
            xx = self._modules[f'Dlinear_{downsample}'](xx)
            
            y += xx
        return y
        # for name, module in self.named_children():
        #     xx=x[:,-:,:]
        #     if name.startswith('downsample'):
        #         x = module(x)
        #     elif name.startswith('Dlinear'):
        #         x = module(x)
        # return x

