'''
@bg
TODO:
- documentation 
'''
import pandas as pd

RAW_IMAGE_FNAME_AS_ID = 'image_name'
RAW_IMAGE_FPATH = 'image_fpath'

CONDITION_COL = 'condition'

NORMAL_VS_NOT_COL = 'is_normal'
VAL_NORMAL = 'Normal'
VAL_NOT_NORMAL = 'Not Normal'

LEFT_OR_RIGHT_EYE_COL = 'is_left_or_right_eye' 
VAL_LEFT_EYE = 'Left Eye'
VAL_RIGHT_EYE = 'Right Eye'

ONH_OR_MACULA_CENTERED_COL = 'is_onh_or_macula_centered'
VAL_ONH_CENTERED = 'ONH'
VAL_MACULA_CENTERED = 'MAC'

def update_fpath(df, fname_col, fp_col, fp_ext, new_dirpath):
    update_fp = lambda x: f"{new_dirpath}/{x}.{fp_ext}"
    df[fp_col] = df[fname_col].apply( update_fp )
    return df 
    
def load_recordz_file(fpath, is_xlsx=False, load_paramz={}):
    df = pd.read_excel( fpath, **load_paramz) if is_xlsx else pd.read_csv( fpath, **load_paramz )
    return df
