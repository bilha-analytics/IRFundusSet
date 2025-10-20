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
    local_dir = Path(local_dir) / "eyepacs_preprocess/eyepacs_preprocess"
    
    recz = [] 
    recz_colz = [RAW_IMAGE_FNAME_AS_ID, RAW_IMAGE_FPATH, ]    
    if local_dir.exists() and local_dir.is_dir(): 
        for fp in local_dir.glob("*.jpeg"):
            fname = fp.stem  
            if fp.exists() and fp.is_file(): 
                recz.append( [fname, str(fp.resolve()), ])    
    
    df = pd.DataFrame.from_records(recz)
    if len(df)>0:
        df.columns = recz_colz
    return df 