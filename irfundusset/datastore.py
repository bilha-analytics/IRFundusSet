'''
@bg
app data store
- statics that come with the app 
    - internal meta and config files: cohorts meta, curated_master_listing csv
- specific to user request 
    - per cohort data source curated csv, b4 stats, 
    - unified dataset harmonized csv, post-harmonize stats, 

Actions:
- file io
- serializable mixin 
- collections management: catalogue, contains, get, 

TODO:
- documentation 
'''
from collections import namedtuple, defaultdict
from pathlib import Path
import pickle 

import pandas as pd 
import numpy as np
import skimage.io as skio 

IS_TEST_MODE = True

_DSTORE_DIR =(Path(__file__).parent/".irfundus_dataset").resolve()   ## TODO:  consider if shd set to user home for user specific runs 
_DSTORE_DIR.mkdir(parents=True, exist_ok=True) 
_DSTORE_CATALOGUE_FPATH = _DSTORE_DIR / "fmazement"
_DSTORE_CATALOGUE = None

def load_dstore():
    global _DSTORE_CATALOGUE
    if _DSTORE_CATALOGUE is None:
        if _DSTORE_CATALOGUE_FPATH.exists() and _DSTORE_CATALOGUE_FPATH.is_file():
            with _DSTORE_CATALOGUE_FPATH.open('rb') as fd: 
                df = pickle.load(fd) 
        else:            
            df = {} 
        _DSTORE_CATALOGUE = df 
    return _DSTORE_CATALOGUE


def add_to_store(iname, ival_url, ival_data):        
    is_pdframe = isinstance(ival_data, pd.DataFrame) 
    _DSTORE_CATALOGUE[iname] = (ival_url, is_pdframe) 
    # i. save content 
    ival_url = _DSTORE_DIR / ival_url 
    ival_url.parent.mkdir(parents=True, exist_ok=True)
    if is_pdframe:
         _ = ival_data.to_csv(ival_url,index=False) if IS_TEST_MODE else ival_data.to_pickle(ival_url,) 
    else:        
        with ival_url.open('wb') as fd:
            pickle.dump(ival_data, fd) 
    
    # ii. update catalogue 
    with _DSTORE_CATALOGUE_FPATH.open('wb') as fd:  
        pickle.dump(_DSTORE_CATALOGUE, fd) 
        
def get_item(iname):
    ival = None
    iurl = _DSTORE_CATALOGUE.get(iname, None) 
    if iurl is not None:        
        iurl, is_pdframe = iurl
        iurl = _DSTORE_DIR / iurl 
        if iurl.exists() and iurl.is_file():
            if is_pdframe:
                ival = pd.read_csv(iurl) if IS_TEST_MODE else pd.read_pickle(iurl)
            else:
                with iurl.open('rb')  as fd:
                    ival = pickle.load(fd) 
    return ival         

def contains_item(iname):
    iurl = _DSTORE_CATALOGUE.get(iname, None) 
    if iurl is None:
        o_ = False
    else:
        iurl = _DSTORE_DIR / iurl[0]  
        o_ = (iurl.exists() and iurl.is_file()) 
    return o_ 

