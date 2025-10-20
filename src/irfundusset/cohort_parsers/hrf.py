'''
@bg
TODO:
- documentation 
'''
from pathlib import Path
import pandas as pd
import numpy as np

from irfundusset.cohort_parsers.commonz import *       ## TODO: this is nasty!!

def create_listing(local_dir):     
    df = pd.DataFrame()
    local_dir = Path(local_dir)  / "images"
    
    recz = [] 
    recz_colz = [RAW_IMAGE_FNAME_AS_ID, RAW_IMAGE_FPATH, ]  
    for fp in local_dir.glob("*.jpg"):
        fname = fp.stem  
        if fp.exists() and fp.is_file(): 
            recz.append( [fname, str(fp.resolve()), ])    
    
    df = pd.DataFrame.from_records(recz)
    df.columns = recz_colz
    return df 