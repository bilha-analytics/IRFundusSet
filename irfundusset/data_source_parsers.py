'''
@bg
parser logics for various datasources
@input = local dir to parse
@output = pd dataframe of  VAR_COL_IMAGE_ID as relative path, image_fpath as full path, 
TODO:
- sync cohort name variables with data_source.py module 
- builder Vs datastruct ????
- documentation 
'''
import pandas as pd 
import numpy as np 
from pathlib import Path
import functools

from irfundusset.cohort_parsers import (idrid as pidrid,
                                              stare as pstare,
                                              chasedb as pchase,
                                              kaggle1000 as pkaggle1000,
                                              papila as ppapila,
                                              hrf as phrf,
                                              five as pfive,
                                              odrid as podrid,
                                              cataracts as pcataracts,
                                              eyepacs as peyepacs
                                              )

from irfundusset.config import IS_TEST

## TODO: move to shared file AND sync with method names below AND better descriptive names e.g. kaggle Vs kaggle1000 Vs kaggle39 
VAR_COL_RELATIVE_FPATH_IS_ID = 'relative_image_fpath'
VAR_COL_IMAGE_FPATH = 'image_fpath'
_cohort_namez = {
    'eyepacs': 'EyePACS',
    'idrid' : 'iDRID',
    'stare' : 'STARE',
    'chasedb' : 'CHASEDB1',
    'kaggle' : 'KAGGLE39',
    'papila' : 'PAPILA',
    'hrf' : 'HRF',
    'five' : 'FIVE',
    'odrid' : 'ODRID',
    'cataracts' : 'RETINA_Cataracts',
}


def tester_df(sxn):
    _old_root_dir = "D:\\xData\\fundus\\"
    
    def update_fpath(rec):        
        rec = rec.replace('/',"\\").replace(_old_root_dir, "")  
        rec = "\\".join(str(rec).split("\\")[3:] )
        return rec 
    
    df = pd.read_csv("D:\\xData\\fundus\\00_HARMONIZED\\10_eye_centered_norm_anom_dataset__harmonized_512.csv")
    df = df[df['listing']==_cohort_namez[sxn]].filter(['image_fpath',])
    return df 

_REGISTRY = {}
def is_parser(func):    
    _REGISTRY[func.__name__] = func
    return func 

def relative_image_pathed(func):
    @functools.wraps(func) 
    def do_func(local_dir, *argz, **kwargz):
        df = func(local_dir, *argz, **kwargz)    
        
        if IS_TEST:
            df.to_csv("./tmp_chk.csv",  index=False)
        df = update_image_id_is_relative_fpath(df, local_dir).filter([VAR_COL_RELATIVE_FPATH_IS_ID,])
        if IS_TEST:
            df = df.sample(13).reset_index(drop=True) 
        return df
    return do_func
 
def get_parser(pname):
    return _REGISTRY.get(pname, None) ## TODO

def update_image_id_is_relative_fpath(df, local_root_dir):   ## TODO: refactor  + Kaggle dir structure match src 
    local_root_dir = str(Path(local_root_dir).resolve())     
    def update_fpath(rec):        
        rec = rec.replace('/',"\\").replace(local_root_dir, "")[1:] 
                
        if 'KAGGLE39' in local_root_dir:                     
            rec = "1000images\\"+rec
        
        return rec 
    if len(df) > 0:
        df[VAR_COL_RELATIVE_FPATH_IS_ID] = df[VAR_COL_IMAGE_FPATH].apply(update_fpath)
    return df 

@is_parser
@relative_image_pathed
def stare(local_dir,):
    df = pstare.create_listing(local_dir)
    return df

@is_parser
@relative_image_pathed
def eyepacs(local_dir,):
    df = peyepacs.create_listing(local_dir)
    return df
        
@is_parser
@relative_image_pathed
def odrid(local_dir,):
    df = podrid.create_listing(local_dir)
    return df

@is_parser
@relative_image_pathed
def idrid(local_dir,):
    df = pidrid.create_listing(local_dir)
    return df

@is_parser
@relative_image_pathed
def chasedb(local_dir,):
    df = pchase.create_listing(local_dir)
    return df

@is_parser
@relative_image_pathed
def kaggle(local_dir,):
    df = pkaggle1000.create_listing(local_dir)
    return df

@is_parser
@relative_image_pathed
def papila(local_dir,):
    df = ppapila.create_listing(local_dir)
    return df

@is_parser
@relative_image_pathed
def hrf(local_dir,):
    df = phrf.create_listing(local_dir)
    return df

@is_parser
@relative_image_pathed
def five(local_dir,):
    df = pfive.create_listing(local_dir)
    return df

@is_parser
@relative_image_pathed
def cataracts(local_dir,):
    df = pcataracts.create_listing(local_dir)
    return df