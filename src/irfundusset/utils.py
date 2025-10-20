'''
@bg
various 
TODO: 
- documentation 
'''
import time
import os 
import configparser

import shutil
from pathlib import Path 

import numpy as np
import cv2

from skimage import img_as_ubyte  
import skimage.exposure as skexpoz
import skimage.io as skio
import skimage.transform as sktrans 

def get_ini_file(fp):
    c = configparser.ConfigParser() 
    c.read_file(open(fp, 'r')) 
    return c 

def update_ini_file( section_variable_value_ls, conf, conf_fpath):
    for section, variable, value in section_variable_value_ls:
        conf.set(section, variable, value) 
    with open(conf_fpath, 'w') as fd:
        conf.write(fd)         
    return conf 


def del_dir(ddir):
    try:
        shutil.rmtree(ddir, ignore_errors=True)
    except Exception as e:
        pass 
                

##====== 
## Image utils 
## TODO: skimage and cv2 --- use one=skimage; avoid unnecessary requirements during install; 
##=====

def normalize_to_distribution(im, center_chanz, dispersion_chanz, weight=1):
    apply_normalize = lambda x, center, dispersion: (x - (weight*center))/((weight*dispersion) + (0 if (weight*dispersion) !=0 else 1e-26))         
    def do_a_channel(cim, chan=0):         
        center = center_chanz[chan]
        dispersion = dispersion_chanz[chan]
        cim = apply_normalize(cim, center, dispersion) 
        return cim 
                
    chanz = im.shape[-1] if len(im.shape)==3 else None 
    if chanz is not None:
        im = np.dstack([do_a_channel(im[:, :, c], chan=c) for c in range(chanz)])   
    else:
        im = do_a_channel(im) 
    im = range_norm_to_ubyte(im)
    return im

def range_norm(im):
    return (im - im.min())/ (im.max() - im.min() + (1e-26 if im.max()<=im.min() else 0) )

def range_norm_to_ubyte(im):
    return img_as_ubyte(range_norm(im))
            

def resize_image(x, size):
    size = size if isinstance(size, (tuple, list)) else (size, size)
    resized = lambda xc: sktrans.resize(xc, size, anti_aliasing=True)
    if size is not None:
        chanz = x.shape[-1] if len(x.shape)==3 else None
        if chanz:
            x = np.dstack([resized(x[:, :, c]) for c in range(chanz)])
        else:
            x = resized(x)
        x = range_norm_to_ubyte(x) ## avoid lossy conversion
    return x

def get_image(fp, size=None):
    return resize_image( skio.imread(fp), size) 

def preprocess_to_clean(im, cl=.01, channel_wise=True):  ## TODO: set best practice values AND/OR allow user to define these
    clean_it = lambda x: skexpoz.equalize_adapthist(x, clip_limit=cl, kernel_size=None) 
         
    if channel_wise and len(im.shape) != 2:
        chanz = im.shape[-1]
        im = np.dstack([clean_it(im[:, :, c]) for c in range(chanz)])
    else:
        im = clean_it(im) 
    return im 

def tight_fit_fov(im, clip=.90, expad=0, clipthresh=10, is_stare=False):
    assert (clip >= 0. ) and (clip <= 1.), f"clip={clip} is not valid. Consider range [0,1]"

    def stare_prep_fov_fit(im, marg = 20 ):
        h, w = im.shape[:2]
        im = im[marg+5:(h-(marg*2)), marg+5:(w-(marg*2)), :] 
        return range_norm_to_ubyte(im)

    def get_fov(xi):     
        size = None
        bi = range_norm_to_ubyte((xi[:, :, 1].copy()))
        o = np.zeros_like(bi)
        
        ## @@19/11 stare seems a bit diff; so update to increase FOV-find chances <<< old blur=9, old min_thresh=30 
        bi = cv2.medianBlur((bi), 39) 
        cz = cv2.threshold((bi), clipthresh, 255, cv2.THRESH_BINARY)[1]
        
        cntz = cv2.findContours(cz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntz = cntz[0] if len(cntz)==2 else cntz[1] 
        if (cntz is not None) and (len(cntz) >0):
            ring = sorted(cntz, key=cv2.contourArea)[-1] 
            (x,y), r = cv2.minEnclosingCircle( ring )
            cv2.circle(o, (int(x),int(y)), int(r-expad), 255, -1)
            size = (x,y), r
            found_fov = True 
        else:
            size = (o.shape[1]//2, o.shape[0]//2), o.shape[0]//2
            found_fov = False
        return o, size, found_fov

    def apply_fov(xi):
        xi = stare_prep_fov_fit(xi) if is_stare else xi 
        fov, fsize, found_fov = get_fov(xi) 
        xi = cv2.bitwise_and(xi, xi, mask=fov) 
        return fov, fsize, xi, found_fov

    def clip_to_fit(xi, fsize):        
        h, w = xi.shape[:2]
        c = None if len(xi.shape)==2 else xi.shape[-1] 
        cx, cy, r = int(fsize[0][0]), int(fsize[0][1]), fsize[1]
        new_r = int(r * clip)
        
        def find_pre_xy():
            pre_w, pre_h = max(0, (cx-new_r)), max(0, (cy-new_r))
            post_w, post_h = pre_w, pre_h
            for i in range(cx):
                if xi[cy, i, 0] >= clipthresh:
                    pre_w = i
                    break
            for i in range(cy):
                if xi[i, cx, 0] >= clipthresh:
                    pre_h = i
                    break
            for i in list(range(cx, w))[::-1]:
                if xi[cy, i, 0] >= clipthresh:
                    post_w = w - i
                    break
            for i in list(range(cy, h))[::-1]:
                if xi[i, cx, 0] >= clipthresh:
                    post_h = h - i
                    break
            return pre_w, pre_h, post_w, post_h
        
        pre_w, pre_h, post_w, post_h = find_pre_xy() 
        new_w, new_h = w - (pre_w + post_w), h - (pre_h + post_h) 
        
        def clipit(ci):
            return ci[pre_h:pre_h+new_h, pre_w:pre_w+new_w] 
        
        xi = np.dstack([clipit(xi[:, :, ci]) for ci in range(c)]) 
        return xi
    
    xi = range_norm_to_ubyte(im.copy())  
    fov, fsize, xi, found_fov = apply_fov(xi) 
    xi = range_norm_to_ubyte(clip_to_fit(xi, fsize)) 
    return fov, xi, found_fov
