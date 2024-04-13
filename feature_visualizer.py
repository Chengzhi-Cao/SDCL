import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
from torchvision.utils import save_image
import cv2

class FeatureVisualizer:
    def __init__(self, size=(128, 64), cmap_type='jet', reduce_type='mean', interpolate_mode='bilinear'):
        self.color_map = plt.get_cmap(cmap_type)
        self.reduce_type = reduce_type
        self.size = size
        self.interpolate_mode = interpolate_mode
        self.height = size[0]
        self.width = size[1]
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.t_mean = torch.Tensor([[[[0.485]], [[0.456]], [[0.406]]]])
        self.t_std = torch.Tensor([[[[0.229]], [[0.224]], [[0.225]]]])

    @staticmethod
    def _check_dimension(x):
        assert x.dim() == 3 or x.dim() == 4, 'Input should be 3D or 4D tensor.'
    
    @staticmethod
    def _normalize(x, mode='spatial', p=2):
        assert x.size(0) == 1
        if mode == 'minmax':
            max_val, min_val = torch.max(x), torch.min(x)
            x = (x - min_val) / (max_val - min_val)
        elif mode == 'spatial':
            x = x.div(torch.norm(x.flatten(), p=p, dim=0))
        elif mode == 'abs':
            x = torch.abs(x)
        return x


    def _recover_numpy(self, x):
        x = self.std * x + self.mean
        x = x * 255.0
        x = np.clip(x, 0, 255)
        x = x.astype(np.uint8)
        return x

    def _recover_torch(self, x):
        if x.is_cuda:
            x = x.detach().cpu()
        return x * self.t_std + self.t_mean

    def _transform_image(self, x, recover):
        """
        Transform image from torch.tensor to numpy.array.
        """
        # reduce by batch, choose the first sample
        x = x[0].unsqueeze(dim=0) if x.dim() == 4 else x.unsqueeze(dim=0)
        x = F.interpolate(x, size=self.size, mode=self.interpolate_mode, align_corners=False)
        x = x.squeeze(dim=0).detach().cpu().numpy().transpose((1, 2, 0))
        return self._recover_numpy(x) if recover else x

    def _transform_feature(self, f):
        """
        Transform feature from torch.tensor to numpy.array.
        """
        # reduce by batch, choose the first sample
        f = f[0].unsqueeze(dim=0) if f.dim() == 4 else f.unsqueeze(dim=0)
        # normalize to [0, 1]
        f = self._normalize(f)
        if self.reduce_type == 'mean':
            f = f.mean(dim=1, keepdim=True)  # reduce by channel
            f = F.interpolate(f, size=self.size, mode=self.interpolate_mode, align_corners=False)
        else:
            raise NotImplementedError
        return f.squeeze().detach().cpu().numpy()

    def _draw(self, x, ca):
        plt.xticks([])  # 去掉x轴
        plt.yticks([])  # 去掉y轴
        plt.axis('off')  # 去掉坐标轴
        ca.imshow(x, cmap=self.color_map)

    def save_feature(self, f, save_path, show=False):


        f = f.pow(2)

        self._check_dimension(f)
        f = self._transform_feature(f.cpu())
        self._draw(f, plt.gca())
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        if show:
            plt.show()
        plt.close()

        _img = cv2.imread(save_path)
        _img = cv2.resize(_img,(self.width,self.height))
        cv2.imwrite(save_path,_img)


    def save_image(self, img, save_path, n_row=8, recover=False):
        self._check_dimension(img)
        if recover:
            img = self._recover_torch(img)
        save_image(img, save_path, nrow=n_row)


    def save_image_single(self, img, save_path, n_row=8, recover=False):
        self._check_dimension(img)
        if recover:
            img = self._recover_torch(img)
        # save_image(img[0], save_path, nrow=n_row)
        save_image(img, save_path, nrow=n_row)

    def save_both(self, img, f, save_path, recover=False, show=False):
        fig = plt.figure()
        ax0 = fig.add_subplot(121, title="image")
        ax1 = fig.add_subplot(122, title="feature")
        self._draw(self._transform_image(img, recover), ax0)
        self._draw(self._transform_feature(f), ax1)
        plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

