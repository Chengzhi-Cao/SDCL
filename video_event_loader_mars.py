from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import functools
from torchvision import transforms as T
import transforms as T
import sys

import torch
from torch import Tensor
from torch.utils.data import Dataset
import random
import torchvision.utils as vutil

import scipy.io as scio

# print(1)
# sys.setrecursionlimit(1000000000)

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

###############################################################################
###############################################################################
def read_mat(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
    
            event_sequence = scio.loadmat(img_path)

            start_time=event_sequence['start_timestamp']
            end_time= np.array(event_sequence['section_event_timestamp'][-1][-1])
            end_time = np.expand_dims(end_time,axis=0)
            end_time = np.expand_dims(end_time,axis=0)
            event_time=event_sequence['section_event_timestamp']
            event_polar=event_sequence['section_event_polarity']
            event_y=event_sequence['section_event_y']
            event_x=event_sequence['section_event_x']
    
            event_frame = np.zeros([3,256,128],int) 

            total_time = end_time - start_time
            time_crop = total_time // 3 + 1
            time_init_num = 1
            for event_i in range(0,event_time.shape[1]):
                if event_time[0,event_i] < time_crop*time_init_num + start_time:
                    if event_polar[0,event_i]>0:
                        event_frame[int(time_init_num-1),event_y[0,event_i]-1,event_x[0,event_i]-1] = event_frame[int(time_init_num-1),event_y[0,event_i]-1,event_x[0,event_i]-1]+1
                    else:
                        event_frame[int(time_init_num-1),event_y[0,event_i]-1,event_x[0,event_i]-1] = event_frame[int(time_init_num-1),event_y[0,event_i]-1,event_x[0,event_i]-1]-1
                else:
                    time_init_num = time_init_num + 1
                
            event_array=np.array(event_frame).astype(np.float32)    # [3,128,64]
            event_array = event_array.transpose((2,1,0))
            img_PIL = Image.fromarray(np.uint8(event_array))

            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img_PIL
###############################################################################
###############################################################################

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

####################################################################################
####################################################################################
####################################################################################
def get_default_event_list_loader():
    image_loader = get_default_event_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def get_default_event_loader():
        return event_loader

def event_loader(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    # print('img_path=',img_path)



    event_sequence = scio.loadmat(img_path)

    start_time=event_sequence['start_timestamp']
    end_time= np.array(event_sequence['section_event_timestamp'][-1][-1])
    end_time = np.expand_dims(end_time,axis=0)
    end_time = np.expand_dims(end_time,axis=0)
    event_time=event_sequence['section_event_timestamp']
    event_polar=event_sequence['section_event_polarity']
    event_y=event_sequence['section_event_y']
    event_x=event_sequence['section_event_x']


    event_frame = np.zeros([3,256,128],int) 

    total_time = end_time - start_time
    time_crop = total_time // 3 + 1
    time_init_num = 1
    for event_i in range(0,event_time.shape[1]):
        if event_time[0,event_i] < time_crop*time_init_num + start_time:
            if event_polar[0,event_i]>0:
                event_frame[int(time_init_num-1),event_y[0,event_i]-1,event_x[0,event_i]-1] = event_frame[int(time_init_num-1),event_y[0,event_i]-1,event_x[0,event_i]-1]+1
            else:
                event_frame[int(time_init_num-1),event_y[0,event_i]-1,event_x[0,event_i]-1] = event_frame[int(time_init_num-1),event_y[0,event_i]-1,event_x[0,event_i]-1]-1
        else:
            time_init_num = time_init_num + 1
        
    event_array=np.array(event_frame).astype(np.float32)    # [3,128,64]
    event_array = event_array.transpose((2,1,0))
    img_PIL = Image.fromarray(np.uint8(event_array))


    return img_PIL

####################################################################################
####################################################################################
####################################################################################
def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video

def imge_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def pil_loader(path):
    with open(path,'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def produce_out(imgs_path,seq_len, stride):
    img_len = len(imgs_path)
    frame_indices = list(range(img_len))
    rand_end = max(0, img_len - seq_len * stride  -1)
    begin_index = random.randint(0, rand_end)
    end_index = min(begin_index + seq_len * stride, img_len)
    indices = frame_indices[begin_index:end_index]
    re_indices= []
    for i in range(0, seq_len * stride, stride):
        add_arg = random.randint(0, stride-1)
        re_indices.append(indices[i + add_arg])
    re_indices = np.array(re_indices)

    out = []
    for index in re_indices:
        out.append(imgs_path[int(index)])
    return out


#################################################################################
#################################################################################
class Video_Event_Dataset_mars(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset,dataset_event, seq_len=15, sample='evenly',
                 transform=None, max_seq_len=200, dataset_name="mars",
                 get_loader = get_default_video_loader,
                 ):

        self.dataset = dataset
        self.dataset_event = dataset_event
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.dataset_name = dataset_name
        self.loader = get_loader()

        self.loader_event = get_default_event_list_loader()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        event_paths, _, _ = self.dataset_event[index]

        num = len(img_paths)

        if self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            frame_indices = list(range(num))
            interval = num // self.seq_len
            indices_list=[]

            if num > self.seq_len:
                for index in range(interval):
                    indices_list.append(frame_indices[index : index+interval * self.seq_len : interval])
            else:
                last_seq = frame_indices[0:]
                for index in last_seq:
                    if len(last_seq) >= self.seq_len:
                        break
                    last_seq.append(index)
                indices_list.append(last_seq)

            imgs_list=[]
            events_list = []
            for indices in indices_list:
                imgs = []
                events = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    event_path = event_paths[index]

                    img = read_image(img_path)
                    _event = read_mat(event_path)

                    if self.transform is not None:
                        img = self.transform(img)
                        _event = self.transform(_event)
                    _event = _event.unsqueeze(0)
                    img = img.unsqueeze(0)
                    events.append(_event)
                    imgs.append(img)

                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)

                events = torch.cat(events, dim=0)
                events_list.append(imgs)
            
            if len(imgs_list) > self.max_seq_len:
                sp = int(random.random() * (len(imgs_list) - self.max_seq_len))
                ep = sp + self.max_seq_len
                imgs_list = imgs_list[sp:ep]
            imgs_array = torch.stack(imgs_list)

            if len(events_list) > self.max_seq_len:
                sp = int(random.random() * (len(events_list) - self.max_seq_len))
                ep = sp + self.max_seq_len
                events_list = events_list[sp:ep]
            events_array = torch.stack(events_list)

            imgs_array = torch.cat((imgs_array,events_array),0)

            if imgs_array == None:
                imgs_array = torch.zeros((16,3,128,64))


            if imgs_array.shape != [16,3,128,64]:
                clip_num = imgs_array.shape[0]
                create_clip = torch.zeros((16-clip_num,3,128,64))
                imgs_array = torch.cat((imgs_array,create_clip))


            return imgs_array, pid, camid, img_paths[0]



#################################################################################
#################################################################################
        elif self.sample == 'Begin_interval':
            event_paths = list(event_paths)

            img_paths = list(img_paths)
            interval = self.seq_len
            num = self.seq_len - 1

            # event
            if len(event_paths) >= interval * num + 1:
                end_index = interval * num + 1
                out = event_paths[0:end_index:interval]

            elif len(event_paths) >= int(interval/2) * num + 1:
                end_index = int(interval/2) * num + 1
                out = event_paths[0:end_index:int(interval/2)]

            elif len(event_paths) >= int(interval/4) * num + 1:
                end_index = int(interval/4) * num + 1
                out = event_paths[0:end_index:int(interval/4)]

            elif len(event_paths) >= int(interval/8) * num + 1:
                end_index = int(interval/8) * num + 1
                out = event_paths[0:end_index:int(interval/8)]

            else:
                out = event_paths[0:interval]
                while len(out) < interval:
                    for index in out:
                        if len(out) >= interval:
                            break
                        out.append(index)

            clip_event = self.loader_event(out)

            if len(clip_event) != 8:
                clip_event = []
                for s in range(8):
                    a = np.zeros((128, 256, 3))
                    a = np.clip(a, 0, 1) 
                    a = (a * 255).astype(np.uint8) 
                    im = Image.fromarray(a)
                    clip_event.append(im)

            if self.transform is not None:
                clip_event = [self.transform(img) for img in clip_event]

            if clip_event == None or len(clip_event) != 8:
                clip_event = torch.zeros((8,3,128,64))
            else:
                clip_event = torch.stack(clip_event, 0)
            
##########################################################################################
##########################################################################################
##########################################################################################
            # img
            if len(img_paths) >= interval * num + 1:
                end_index = interval * num + 1
                out = img_paths[0:end_index:interval]

            elif len(img_paths) >= int(interval/2) * num + 1:
                end_index = int(interval/2) * num + 1
                out = img_paths[0:end_index:int(interval/2)]

            elif len(img_paths) >= int(interval/4) * num + 1:
                end_index = int(interval/4) * num + 1
                out = img_paths[0:end_index:int(interval/4)]

            elif len(img_paths) >= int(interval/8) * num + 1:
                end_index = int(interval/8) * num + 1
                out = img_paths[0:end_index:int(interval/8)]

            else:
                out = img_paths[0:interval]
                while len(out) < interval:
                    for index in out:
                        if len(out) >= interval:
                            break
                        out.append(index)

            clip = self.loader(out)
            if self.transform is not None:
                clip = [self.transform(img) for img in clip]

            clip = torch.stack(clip, 0)

            clip = torch.cat((clip,clip_event),0)
            # print('clip=',clip.shape)   # [16,3,128,64]


            if clip == None:
                clip = torch.zeros((16,3,128,64))


            if clip.shape != [16,3,128,64]:
                clip_num = clip.shape[0]
                create_clip = torch.zeros((16-clip_num,3,128,64))
                clip = torch.cat((clip,create_clip))
            
            return clip, pid, camid, out

#################################################################################
#################################################################################

        elif self.sample == 'Random_interval':
            # img
            img_paths = list(img_paths)
            stride = 8

            if len(img_paths) >= self.seq_len * stride :
                new_stride = stride
                out = produce_out(img_paths, self.seq_len, new_stride)

            elif len(img_paths) >= self.seq_len * int(stride/2):
                new_stride = int(stride/2)
                out = produce_out(img_paths, self.seq_len, new_stride)

            elif len(img_paths) >= self.seq_len * int(stride/4):
                new_stride = int(stride/4)
                out = produce_out(img_paths, self.seq_len, new_stride)

            elif len(img_paths) >= self.seq_len * int(stride/8):
                new_stride = int(stride/8)
                out = produce_out(img_paths, self.seq_len, new_stride)

            else:
                index = np.random.choice(len(img_paths), size=self.seq_len,replace=True)
                index.sort()
                out = [img_paths[index[i]] for i in range(self.seq_len)]

            clip = self.loader(out)
            clip = self.transform(clip)
            clip = torch.stack(clip, 0)


            # events
            event_paths = list(event_paths)
            stride = 8

            if len(event_paths) >= self.seq_len * stride :
                new_stride = stride
                out = produce_out(event_paths, self.seq_len, new_stride)

            elif len(event_paths) >= self.seq_len * int(stride/2):
                new_stride = int(stride/2)
                out = produce_out(event_paths, self.seq_len, new_stride)

            elif len(event_paths) >= self.seq_len * int(stride/4):
                new_stride = int(stride/4)
                out = produce_out(event_paths, self.seq_len, new_stride)

            elif len(event_paths) >= self.seq_len * int(stride/8):
                new_stride = int(stride/8)
                out = produce_out(event_paths, self.seq_len, new_stride)

            else:
                index = np.random.choice(len(event_paths), size=self.seq_len,replace=True)
                index.sort()
                out = [event_paths[index[i]] for i in range(self.seq_len)]

            clip_event = self.loader_event(out)
            
            if len(clip_event) != 8:
                clip_event = []
                for s in range(8):
                    a = np.zeros((128, 256, 3))
                    a = np.clip(a, 0, 1) 
                    a = (a * 255).astype(np.uint8)
                    im = Image.fromarray(a)
                    clip_event.append(im)

                
            clip_event = self.transform(clip_event)
            
            if clip_event == None or len(clip_event) != 8:
                clip_event = torch.zeros((8,3,128,64))
            else:
                clip_event = torch.stack(clip_event, 0)

            clip = torch.cat((clip,clip_event),0)


            if clip == None:
                clip = torch.zeros((16,3,128,64))

            if clip.shape != [16,3,128,64]:
                clip_num = clip.shape[0]
                create_clip = torch.zeros((16-clip_num,3,128,64))
                clip = torch.cat((clip,create_clip))

            return clip, pid, camid, out

        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))



