'''
@bg
TODO:
- documentation 
'''
from pathlib import Path
import pandas as pd
import numpy as np

from irfundusset.cohort_parsers.commonz import *       ## TODO: this is nasty!!

IDRIDataset_segz_elements = ["1. Microaneurysms", "2. Haemorrhages", 
                    "3. Hard Exudates", "4. Soft Exudates",
                    "5. Optic Disc"]

IDRIDataset_localize_elements = [("1. Optic Disc Center Location","OD",), 
                    ("2. Fovea Center Location", "Fovea")] 

IDRIDataset_dir_struct = {
        "segment" : {'root':"A. Segmentation", 
                     'train' : {
                         'origi': {'kind': 'img', 
                                   'fpath': "1. Original Images/a. Training Set" },
                         'gtruth': {'kind': 'img', 
                                   'fpath': [f"2. All Segmentation Groundtruths/a. Training Set/{ds}" for ds in IDRIDataset_segz_elements] },
                         },
                     'test' : {
                         'origi': {'kind': 'img', 
                                   'fpath': "1. Original Images/b. Testing Set" },
                         'gtruth': {'kind': 'img', 
                                   'fpath': [f"2. All Segmentation Groundtruths/b. Testing Set/{ds}" for ds in IDRIDataset_segz_elements] },
                         },
                     },
        "grade" : {'root':"B. Disease Grading",
                     'train' : {
                         'origi': {'kind': 'img', 
                                   'fpath': '1. Original Images/a. Training Set'},
                         'gtruth': {'kind': 'csv', 
                                   'fpath': '2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'},
                         },
                     'test' : {
                         'origi': {'kind': 'img', 
                                   'fpath': '1. Original Images/b. Testing Set'},
                         'gtruth': {'kind': 'csv', 
                                   'fpath': '2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv'},
                         },
                     },
    }

def create_listing(local_dir):
    df = pd.DataFrame()
    local_dir = Path(local_dir)

    recz = [] 
    recz_colz = [RAW_IMAGE_FNAME_AS_ID, RAW_IMAGE_FPATH, ] 
    for k, v in IDRIDataset_dir_struct.items():
        droot = v['root']     
        for split in ['train','test']:
            data_origi = v[split]['origi']['fpath']
            ## i. make record for each origi img --> kind is always img,
            dl_origi = (local_dir / droot / data_origi).glob("*")
            for fp in dl_origi:
                fname = fp.stem  
                if fp.exists() and fp.is_file(): 
                    recz.append( [fname, str(fp.resolve()), ])

    df = pd.DataFrame.from_records(recz) 
    df.columns = recz_colz 
    return df

