from .Mars import *
from .Duke import *
from .PRID import *
from .PRID_event import *
from .PRID_event_vid import *
from .Mars_event_vid import *
from .PRID_dark import *
from .Low import *
from .PRID_dark2 import *
from .PRID_dark2_event_vid import *
from .Low_event_vid import *
from .PRID_dark2_SCI import *
from .iLIDSVID import *
from .iLIDSVID_event import *
from .iLIDSVID_event_vid import *
from .PRID_blur9_10 import *
from .PRID_blur19_20 import *
from .PRID_blur29_30 import *
from .iLIDSVID_blur4_5 import *
from .PRID_block import *
from .iLIDSVID_block import *
from .iLIDSVID_event_vid_blur4_5 import *
from .iLIDSVID_block_event import iLIDSVID_block_event
from .PRID_blur_event import PRID_blur_event
from .iLIDSVID_dark import iLIDSVID_dark
from .iLIDSVID_dark_event import iLIDSVID_dark_event
from .Low_SCI import Low_SCI
from .iLIDSVID_dark_SCI import iLIDSVID_dark_SCI
from .PRID_mat_img import PRID_mat_img
from .PRID_mat_img2 import PRID_mat_img2
from .PRID_event_img_vid import PRID_event_img_vid

from .iLIDSVID_dark_LIME import iLIDSVID_dark_LIME
from .PRID_dark2_LIME import PRID_dark2_LIME
from .Low_LIME import Low_LIME
from .Low_event import Low_event
from .iLIDSVID_mat_img import iLIDSVID_mat_img
from .PRID_dark_event_img_vid import PRID_dark_event_img_vid

from .PRIDE import PRIDE
from .PRIDE_event_vid import PRIDE_event_vid

__factory = {
    'pride':PRIDE,
    'pride_event_vid':PRIDE_event_vid,
    'mars': Mars,
    'duke':DukeMTMCVidReID,
    'prid':PRID,
    'prid_event':PRID_event,
    'prid_event_vid':PRID_event_vid,
    'mars_event_vid':Mars_event_vid,
    'prid_dark':PRID_dark,
    'Low':Low,
    'Low_event_vid':Low_event_vid,
    'prid_dark2':PRID_dark2,
    'prid_dark2_SCI':PRID_dark2_SCI,
    'prid_dark2_event_vid':PRID_dark_event_vid,
    'iLIDSVID':iLIDSVID,
    'iLIDSVID_event':iLIDSVID_event,
    'iLIDSVID_event_vid':iLIDSVID_event_vid,
    'PRID_blur9_10':PRID_blur9_10,
    'PRID_blur19_20':PRID_blur19_20,
    'PRID_blur29_30':PRID_blur29_30,
    'iLIDSVID_blur4_5':iLIDSVID_blur4_5,
    'PRID_block':PRID_block,
    'iLIDSVID_block':iLIDSVID_block,
    'iLIDSVID_event_vid_blur4_5':iLIDSVID_event_vid_blur4_5,
    'iLIDSVID_block_event':iLIDSVID_block_event,
    'PRID_blur_event':PRID_blur_event,
    'iLIDSVID_dark':iLIDSVID_dark,
    'iLIDSVID_dark_event':iLIDSVID_dark_event,
    'Low_SCI':Low_SCI,
    'iLIDSVID_dark_SCI':iLIDSVID_dark_SCI,
    'PRID_mat_img':PRID_mat_img,
    'PRID_mat_img2':PRID_mat_img2,
    'PRID_event_img_vid':PRID_event_img_vid,
    
    'iLIDSVID_dark_LIME':iLIDSVID_dark_LIME,
    'PRID_dark2_LIME':PRID_dark2_LIME,
    'Low_LIME':Low_LIME,
    'Low_event':Low_event,
    'iLIDSVID_mat_img':iLIDSVID_mat_img,
    'PRID_dark_event_img_vid':PRID_dark_event_img_vid,


}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)