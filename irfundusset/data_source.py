'''
@bg
Datasources + meta + Unified Source objects
roles: data objects/structures, parser logic @ specific cohorts/listing, 

## TODO: 
- place tqdm calls and avoid too many of them being written to out
- refactor commonz
- remove extras @ post-harmonize analysis material
- documentation 
'''
from tqdm import tqdm  ##TODO: command line Vs notebook

from pathlib import Path 
from collections import namedtuple, defaultdict
from dataclasses import dataclass, field 

import pandas as pd
import numpy as np 
import skimage.io as skio 
import skimage.transform as sktrans

import irfundusset.data_source_parsers as dparsers
import irfundusset.datastore as dstore
import irfundusset.utils as U

from irfundusset.config import IS_DEPLOY

## TODO: move to shared file AND sync with method names below AND better descriptive names e.g. kaggle Vs kaggle1000 Vs kaggle39 
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

## some variable constants
VAR_TMP_DIR = ".tmp_cohortz" 

VAR_OUT_HARMD_MASTER_CSV = lambda im_w: f"irfundus_set_unified_master__{im_w:03d}.csv"

VAR_STORE_HARMD_DSOURCE_DIR = "harmonized_v3"
VAR_STORE_HARMD_DSOURCE_COLLXN_FP = lambda x: f"{x}/collectionz"
VAR_STORE_HARMD_DSOURCE_CSV_FP = lambda x: f"{x}/harmonized_unified_curated.csv"
VAR_STORE_HARMD_DSOURCE_B4_STATS_FP = lambda x: f"{x}/harmonized_unified_curated_b4stats.csv"
VAR_STORE_HARMD_DSOURCE_HARMD_STATS_FP = lambda x: f"{x}/harmonized_unified_curated_harmdstats.csv"

VAR_STORE_COHORTS_DSOURCE_DIR = "cohorts"
VAR_STORE_PER_COHORT_DSOURCE_DIR = lambda x: f"{VAR_STORE_COHORTS_DSOURCE_DIR}/{x}" 
VAR_STORE_PER_COHORT_DSOURCE_CSV_FP = lambda xd: f"{xd}/curated.fdd"
VAR_STORE_PER_COHORT_DSOURCE_B4_STATS_FP = lambda xd: f"{xd}/curated_b4stats.fdd"
VAR_STORE_PER_COHORT_DSOURCE_HARMD_STATS_FP = lambda xd: f"{xd}/curated_harmdstats.fdd"

VALUE_IS_NORMAL = 'Normal'
VALUE_IS_NOT_NORMAL = 'Not normal'
VALUE_IS_UNDEF_NORMAL = np.nan    ## UNDEF

VALUE_ENTIRE_DATASET_LISTING_NAME = 'entire_dataset'

VAR_COL_IS_NORMAL = 'is_normal'
VAR_COL_IMAGE_ID = 'relative_image_fpath'
VAR_COL_LISTING = 'listing'
VAR_COL_HARMD_IMAGE_FPATH = 'image_fpath'
VAR_COL_TMP_IMAGE_FPATH = 'tmp_preproc_image_fpath'
VAR_COL_SRC_IMAGE_EXISTS = 'local_dir_src_image_exists'

VAR_COL_STATZ_N_SAMPLES_ALL = 'n_samples_all'
VAR_COL_STATZ_PERC_SAMPLES_NORMAL = 'perc_samples_normal'

CURATED_MASTER_CSV_FPATH = (Path(__file__).parent/"curated_master_listing.fdd").resolve()    ##TODO: hold variable file_name in common place
## TODO: read these from a shared space 
_dset_presetz_colz = ['image_id', 'listing', 'relative_image_fpath', 'image_name', 'is_normal', 
                'src_condition', 'src_is_normal', 
                'is_left_or_right_eye','is_onh_or_macula_centered', 
                'harmonized_condition_label']  
_dset_harmd_colz = ['image_fpath', 'local_dir_src_image_exists',] 

META_COHORTS_FILE = "meta_cohorts.ini"
META_COHORTS_FPATH = str((Path(__file__).parent/META_COHORTS_FILE).resolve())
CONF_META_COHORTS = U.get_ini_file(META_COHORTS_FPATH)

@dataclass
class DatasourceMeta:
    name: str  = field(init=True)
    primary_conditon: str = field(default=None, init=False)
    web_url: str = field(default=None, init=False)
    citation: str = field(default=None, init=False)
    description: str = field(default=None, init=False)
    license_notes: str = field(default=None, init=False)
    
    def __post_init__(self):
        assert self.name in CONF_META_COHORTS.sections(), \
            f"'{self.name}' is not in available options. Consider {CONF_META_COHORTS.sections()}" 
        _ = [setattr(self, k, v) for k,v in CONF_META_COHORTS.items(self.name)]  
        
class NotGeneratedYetError(Exception):
    def __init__(self, msg=None) -> None:
        msg = "Generate the dataset first" if msg==None else msg
        super().__init__(msg) 
     
class AlreadyGeneratedError(Exception):
    MSG = "Dataset alredy generated and is available. Nothing to do. Use force_generate=True if still want to (re)generate"
    def __init__(self, msg=None) -> None:
        msg = self.MSG if msg==None else msg
        super().__init__(msg) 
        
class DataframeDataset:
    target_colz = [VAR_COL_IS_NORMAL, 'src_condition', 'src_is_normal', 'harmonized_condition'] 
    other_colz = [VAR_COL_LISTING, VAR_COL_HARMD_IMAGE_FPATH, 'is_left_or_right_eye', 'is_onh_or_macula_centered',] 
    col_image_id = VAR_COL_HARMD_IMAGE_FPATH 
    _listing = None
    
    def __init__(self, csv, xtransform=None, ytransform=None, target_col=None):
        assert (target_col is None or target_col in self.target_colz),\
            f"'{target_col}' is not recognized. Available options are {self.target_colz}"
        self.xtransform = xtransform 
        self.ytransform = ytransform 
        self.target_col = VAR_COL_IS_NORMAL if target_col is None else target_col
        self._init_listing(csv) 
        
    def _init_listing(self, csv):
        # i. load listing & drop NaN @ target col
        df_all = csv if isinstance(csv, pd.DataFrame) else pd.read_pickle(csv)
        df_all = df_all.filter(self.other_colz+self.target_colz).dropna(subset=self.target_col).reset_index(drop=True)   	
        # ii. subset normalz
        df_normz = df_all[df_all[VAR_COL_IS_NORMAL]==VALUE_IS_NORMAL]
        # iii. subset anomz to n match normalz 
        df_anomz = df_all[df_all[VAR_COL_IS_NORMAL]==VALUE_IS_NOT_NORMAL]
        df_anomz = df_anomz.sample(min(len(df_anomz), len(df_normz))).reset_index(drop=True) 
        # iv. fini
        self._listing = pd.concat([df_normz, df_anomz])
    
    def __len__(self):
        return len(self._listing)  if self._listing is not None else 0
    
    def __getitem__(self, ix):
        xi = self._listing.iloc[ix].to_dict() 
        
        x = skio.imread(xi[VAR_COL_HARMD_IMAGE_FPATH]) 
        if self.xtransform is not None:
            x = self.xtransform(x) 
        
        y = xi[self.target_col]  
        if self.ytransform is not None:
            y = self.ytransform(x) 
            
        xi['target'] = y
        xi['image'] = x
        return xi
    
    @property 
    def target_distribution(self):
        if self._listing is None:
            return None
        return self._listing[self.target_col].value_counts() 
        
class SaveableSource:
    name = None
    is_cohort = True
    
    @property
    def curated_listing_exists(self):
        return dstore.contains_item(self.curated_listing_key) 
    
    @property
    def curated_listing_key(self):
        return f"{self.name}__listing"
    
    @property
    def curated_b4_harmonize_stats_key(self):
        return f"{self.name}__b4_istatz"
    
    @property
    def curated_after_harmonize_stats_key(self):
        return f"{self.name}__after_istatz"
    
    @property
    def collection_key(self):
        return f"{self.name}__collection"
    
    @property
    def collection_obj(self):
        return dstore.get_item(self.collection_key) 
    
    @property
    def curated_csv(self):
        return dstore.get_item(self.curated_listing_key) 
    
    @property
    def b4_istatz(self):
        return dstore.get_item(self.curated_b4_harmonize_stats_key)
    
    @property
    def after_istatz(self):
        return dstore.get_item(self.curated_after_harmonize_stats_key) 
        
    def persist(self, df_curated=None, b4istatz=None, a5istatz=None, collexn=None):  
        cdir = (VAR_STORE_PER_COHORT_DSOURCE_DIR(self.name) if self.is_cohort else self.name)
        # csv_fp, b4istats_fp, a5istats_fp = None, None, None 
        if df_curated is not None:            
            csv_fp = (VAR_STORE_PER_COHORT_DSOURCE_CSV_FP if self.is_cohort else VAR_STORE_HARMD_DSOURCE_CSV_FP)(cdir)
            dstore.add_to_store(self.curated_listing_key, csv_fp, df_curated) 
            
        if b4istatz is not None:
            b4istats_fp = (VAR_STORE_PER_COHORT_DSOURCE_B4_STATS_FP if self.is_cohort else VAR_STORE_HARMD_DSOURCE_B4_STATS_FP)(cdir)
            dstore.add_to_store(self.curated_b4_harmonize_stats_key, b4istats_fp, b4istatz)  
            
        if a5istatz is not None:
            a5istats_fp = (VAR_STORE_PER_COHORT_DSOURCE_HARMD_STATS_FP if self.is_cohort else VAR_STORE_HARMD_DSOURCE_HARMD_STATS_FP)(cdir)
            dstore.add_to_store(self.curated_after_harmonize_stats_key, a5istats_fp, a5istatz) 
            
        if (collexn is not None) and (not self.is_cohort):
            collexn_fp = VAR_STORE_HARMD_DSOURCE_COLLXN_FP(cdir)       
            dstore.add_to_store(self.collection_key, collexn_fp, collexn) 
    
    
class CohortDataSource:
    '''
    obj: find, parse, preproc, populate curated, b4stats and status. Prepares content for the unified collection. 
    - cohort specif operations @ parse source directory, preprocess images (fit FOV, resize, )
    '''        
    def __init__(self, sxn_name) -> None:
        self.name = sxn_name
        self.is_cohort = True 
        
    def get_output_tmp_dir(self, out_dir):        
        tmp = Path(out_dir) / VAR_TMP_DIR / self.name
        tmp.mkdir(parents=True, exist_ok=True) 
        return tmp 
    
    def generate_listing(self, local_dir, out_dir, out_img_w, clahe_b4_harmonize=False, ):
        ''' 
        '''
        # i. get curated csv which has labels normal, not normal AND nan filtered out already  
        df_src = dparsers.get_parser(self.name)(local_dir)                                    ## has VAR_COL_IMAGE_ID as relative path, image_fpath as full path, 
        
        if len(df_src) == 0:
            return df_src
        
        df_curated = self._get_curated_df()          
        
        df_curated = pd.merge(df_curated, df_src, how='right', on=VAR_COL_IMAGE_ID)           ## filter curated to subselect this cohorts items only
        df_curated.dropna(subset='src_condition', inplace=True)

        # ii. preprocess @ entire dataset = fit FOV + resize TODO: should we do some basic clahe??
        df_preproc = self._preprocess(df_curated, local_dir, out_dir, out_img_w, clahe_b4_harmonize=clahe_b4_harmonize)               ## retns df <img_id, preproc_tmp_fpath, src_img_exists>      
        if df_preproc is not None:
            df_curated = pd.merge(df_curated, df_preproc, how='left', on=VAR_COL_IMAGE_ID)         ## add preprocessed tmp fpath cols
                
        return df_curated    
    
    def _get_curated_df(self):
        '''return full csv for give cohort; includes curated labels for normal, not normal. 
            - normalize against all @ src_condition is known 
        logic: curated has 3 labels: normal, not normal and np.nan. 
        - If only keep normal and not normal, then harmonize will only consider those. 
        - No need to run on entire universe set b/c should never serve np.nan 
        - QUE TODO: is there a relevant use case for serving np.nan 
        
        > @@27/11 do dropna condition + duplicates : @ per cohort level 
        ''' 
        cname = _cohort_namez[self.name] 
        df = pd.read_pickle(CURATED_MASTER_CSV_FPATH)
        df = df[df[VAR_COL_LISTING]==cname].rename(
                                                columns={VAR_COL_LISTING : f"old_{VAR_COL_LISTING}"}
                                            ).dropna(subset='src_condition'
                                            ).drop_duplicates(subset=VAR_COL_IMAGE_ID
                                            ).reset_index(drop=True)
        return df 
             
            
    def _preprocess(self, df_curated, local_dir, out_dir, out_img_w=32, clahe_b4_harmonize=False):
        '''
        TODO: out_dir Vs a specific temp_dir @ sync with harmonizer 
        obj:
        - foreach img: fit FOV, resize, calc stats 
        '''
        if len(df_curated)==0:
            return None 
        
        def fov_fit(im):
            # i. fov fit 
            _, im, found_fov = U.tight_fit_fov(im, is_stare=('stare' in self.name.lower()))             
            # ii. some cohorts have weird margins; clean those up. <-- NOOP for now b/c = [RFMID,] and RFMID is not in the curated list b/c couldn't find the conditions GT csv from source
            return im, found_fov 
                        
        # i. tmp hold here for b4 harmonize 
        tmp_out = self.get_output_tmp_dir(out_dir)
        
        # ii. cohort specific preprocs @ FOV fit and img resize 
        local_dir = Path(local_dir)
        pbar = tqdm(total=len(df_curated))
        pbar.set_description(f'parsing local dir [{self.name}]') 
        im_recz = []
        for fid in df_curated[VAR_COL_IMAGE_ID]:
            # ii.read it 
            fp = local_dir / fid 
            fexists = fp.exists() and fp.is_file() 
            
            # iii. fov fit 
            im = skio.imread(fp) 
            im, found_fov = fov_fit(im)             
            if not found_fov:
                with open('fov_not_found_files.txt', 'a') as fd:
                    fd.writelines(f"[{self.name}] {str(fid)}\n")
                    
            # # iv. resize  & preprocess iff  
            im = U.resize_image(im, out_img_w)     
            if clahe_b4_harmonize:
                im = U.preprocess_to_clean(im)
            
            # v. tmp save it 
            fp = tmp_out/fid
            fp.parent.mkdir(parents=True, exist_ok=True)
            skio.imsave(fp, U.range_norm_to_ubyte(im)
                        )
            # vi. track and update  
            im_recz.append( [fid, fexists, str(fp.resolve()) if fexists else np.nan]) 
            
            pbar.update(1)
        
        # iii.  build df 
        df_tmp = pd.DataFrame.from_records(im_recz)
        df_tmp.columns = [VAR_COL_IMAGE_ID, VAR_COL_SRC_IMAGE_EXISTS, VAR_COL_TMP_IMAGE_FPATH]        
        pbar.close() 
        
        return df_tmp   

class HarmonizedDataSource(SaveableSource):
    '''
    Obj: 
    - Is a collection of datasources; knows what's contained within the `unified` lot  + does the consolidation logic 
    - Entry point for generate and load/get dataset 
    '''
    def __init__(self, out_dir, out_image_w=32) -> None:
        self.name = f'{Path(out_dir).name}__{out_image_w:03d}'
        self.is_cohort = False 
        self.out_image_w = out_image_w 
        self.out_dir = (Path(out_dir).parent/self.name).resolve()
        dstore.load_dstore()
        self.collection = self.collection_obj
        if self.collection is None:
            self.collection = []
    
    def generate(self, in_cohorts_config, clahe_b4_harmonize=False, method='iqr', force_regenerate=False):
        ''' 
        logic: collect data from a list of cohorts and consolidate into one unified dataset 
        - for cohort in collection as per in_cohorts_cofig, parse dir + create csv that's filtered to curated + source image stats
            - each datasource has cohort specific logic for parse dir + preprocess(fov fit, img resize, drops, etc). So will need out_dir 
            - returns csv to curated_aligned_csv @ labels normal, not normal, undef/nan
        - clean_up at the end << this func here should clear temp dir of images created in preprocess step by cohort data source TODO: rethink that flow 
        - create unified_csv form curated csv(s)  
        - get b4 stats 
        - call harmonizer(unified_csv, cohort stats list, out_dir) -> df[img_fpath, out_fpath, new_harmonized_stats]
        - update unified_csv with harmonized infor + persist to dstore
            .filter(self._dset_colz)
        - get a5/after stats 
        @return
        - generate status, unified_harmonized_csv url 
        '''
        if self.is_generated and not force_regenerate:
            return AlreadyGeneratedError.MSG
                
        out_dir = self.out_dir 
        # i. get cohort data sources
        curated_dfz = []      
        in_config = U.get_ini_file(in_cohorts_config)
        
        pbar = tqdm( total=len(in_config.sections()) + 4)
        
        for sxn in in_config.sections():
            pbar.set_description(f"Parsing {sxn}")
            self.collection.append(sxn) 
            ds = CohortDataSource(sxn_name=sxn) 
            df = ds.generate_listing(local_dir=in_config.get(sxn, 'local_dir'), 
                                                out_dir=out_dir, 
                                                out_img_w=self.out_image_w,
                                                clahe_b4_harmonize=clahe_b4_harmonize) 
            if len(df) > 0:
                df[VAR_COL_LISTING] = sxn 
                curated_dfz.append(df)              
            pbar.set_description(f"Finished Parsing {sxn}")
            pbar.update(1)
            
        uni_curated_df = pd.concat(curated_dfz)
                        
        ## ii. b4 stats df listing = cohort + all 
        pbar.set_description(f"Generating b4 stats")
        df_b4_statz, uni_b4_statz = get_image_stats(uni_curated_df, VAR_COL_TMP_IMAGE_FPATH) 
        df_b4_statz.rename(columns={c:f"b4_{c}" for c in df_b4_statz.columns if c !=VAR_COL_IMAGE_ID}, inplace=True)
        uni_curated_df = pd.merge(uni_curated_df, df_b4_statz, how='left', on=VAR_COL_IMAGE_ID)          
        pbar.set_description(f"Finished generating b4 stats")
        pbar.update(1)
        
        # iii. curated df after harmonize 
        pbar.set_description(f"Harmonizing datasets")
        harmd_df = self._harmonize(uni_curated_df, uni_b4_statz, out_dir, method=method)
        uni_curated_df = pd.merge(uni_curated_df, harmd_df, how='left', on=VAR_COL_IMAGE_ID) 
        pbar.set_description(f"Finished harmonizing datasets")
        pbar.update(1)
        
        # iv. statz after harmonize 
        pbar.set_description(f"Generting after stats")
        df_a5_statz, uni_a5_statz = get_image_stats(uni_curated_df, VAR_COL_HARMD_IMAGE_FPATH) 
        uni_curated_df = pd.merge(uni_curated_df, df_a5_statz, how='left', on=VAR_COL_IMAGE_ID)         
        pbar.set_description(f"Finished generating after stats")
        pbar.update(1)
        
        # v. persist, clean up -- KEEP tmp_cohortz dir for analysis; delete afterwards 
        pbar.set_description(f"Saving and cleaning up")
        if IS_DEPLOY:  ## TODO: better flagging and/or rethink stats approach 
            uni_curated_df = uni_curated_df.filter([c for c in uni_curated_df.columns if not c.startswith('tmp_')])
            U.del_dir(Path(out_dir) / VAR_TMP_DIR) 
                        
        self._save_unified_master_csv(uni_curated_df, out_dir)
        
        self.persist(df_curated=uni_curated_df, 
                     b4istatz=uni_b4_statz, 
                     a5istatz=uni_a5_statz,
                     collexn=self.collection, )
        pbar.set_description(f"Finished generating dataset")
        pbar.update(1)
        
        pbar.close() 
        return "Dataset Generated"
    
    def _save_unified_master_csv(self, df, out_dir):
        '''filter to user colz + save to outdir ; will keep is_normal is NaN as well '''
        csv_fp = Path(out_dir) / VAR_OUT_HARMD_MASTER_CSV(self.out_image_w) 
        
        df = df.filter(_dset_presetz_colz+_dset_harmd_colz 
                       ).rename(columns={'listing':'cohort', 
                                         VAR_COL_IMAGE_ID: 'image_id'}
                        )
                       
        df.to_csv(csv_fp, index=False)
           
            
    def _harmonize(self, curated_df, pre_b4statz_df, out_dir, method='zscore'):
        assert method in (None, 'iqr', 'irange', 'zscore'), f"'{method}' not recognized. Currently supporting robust median with (zscore, irange, iqr) only"  ##TODO: placement
        n_cohort = lambda cht: pre_b4statz_df[pre_b4statz_df[VAR_COL_LISTING]==cht]['n_samples_all'].values[0]         
        n_all = n_cohort(VALUE_ENTIRE_DATASET_LISTING_NAME)
        
        def get_per_channel_stats(b4statz_df, metric, chan, cohort=None):
            prefix_chan = "" if chan is None else (["red","green","blue"][chan]+"_")
            metric = f"{prefix_chan}{metric}"
            cohort = VALUE_ENTIRE_DATASET_LISTING_NAME if cohort is None else cohort
            return b4statz_df[b4statz_df[VAR_COL_LISTING]==cohort][metric].values[0] 
        
        def do_normalize(im, centerz, dispersionz, cht=None):            
            cht_factor = (n_cohort(cht) / n_all) if cht is not None else 1 
            im = U.normalize_to_distribution(im, 
                                             center_chanz=centerz, 
                                             dispersion_chanz=dispersionz, 
                                             weight=cht_factor)  
            return im 
        
        def do_normalize_a_dframe(df_in, b4stats_in, cohort_in=None, is_intermediate_step=True):
            ## i. harmonize statistics     
            if method=='zcore':
                centerz = [get_per_channel_stats(b4stats_in, 'mean', chan=c, cohort=cohort_in) for c in range(3) ]
                dispersionz = [get_per_channel_stats(b4stats_in, 'std', chan=c, cohort=cohort_in) for c in range(3) ]
            else:
                disp_m1, disp_m2 = ('min', 'max') if method=='irange' else ('q1', 'q3')
                centerz = [get_per_channel_stats(b4stats_in, 'median', chan=c, cohort=cohort_in) for c in range(3) ]
                dispersionz = [(get_per_channel_stats(b4stats_in, disp_m2, chan=c, cohort=cohort_in) - \
                                    get_per_channel_stats(b4stats_in, disp_m1, chan=c, cohort=cohort_in)) for c in range(3) ]
            
            ## ii. run it 
            df = df_in.filter([VAR_COL_IMAGE_ID, VAR_COL_TMP_IMAGE_FPATH, VAR_COL_LISTING, VAR_COL_IS_NORMAL])
            harmd_fpz = []
            pbar = tqdm(total=len(df))
            pbar.set_description(f"harmonizing image files [{cohort_in} - tmp={is_intermediate_step}]")
            for _, rec in df.iterrows():
                cohort = rec[VAR_COL_LISTING]                 
                tim_fp = rec[VAR_COL_TMP_IMAGE_FPATH]             
                im = skio.imread(tim_fp)
                im = do_normalize(im, centerz=centerz, dispersionz=dispersionz, cht=None)# if is_intermediate_step else cohort) 
                
                if is_intermediate_step:
                    skio.imsave(tim_fp, im) 
                else:
                    fp = Path(rec[VAR_COL_IMAGE_ID]).suffix
                    fp = str(rec[VAR_COL_IMAGE_ID])[:-len(fp)] 
                    fp = (Path(out_dir) / cohort / f"{fp}.png").resolve()     ## all to same image format 
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    skio.imsave(fp, im) 
                    harmd_fpz.append(str(fp))
                    
                pbar.update(1)
            
            if not is_intermediate_step:
                df = df.filter([VAR_COL_IMAGE_ID, VAR_COL_TMP_IMAGE_FPATH, ])
                df[VAR_COL_HARMD_IMAGE_FPATH]  = harmd_fpz 
            
            pbar.close() 
            return df
        
        ## i. within cohort run & update b4 stats 
        df = [do_normalize_a_dframe(b4stats_in=pre_b4statz_df,
                                    cohort_in=cohort,
                                    df_in=curated_df[curated_df[VAR_COL_LISTING]==cohort], 
                                    is_intermediate_step=True) \
                                        for cohort in curated_df[VAR_COL_LISTING].unique() \
                                            if cohort != VALUE_ENTIRE_DATASET_LISTING_NAME]
        df = pd.concat(df) 
        b4statz_df, _ = get_image_stats(df, VAR_COL_TMP_IMAGE_FPATH, keep_listing=True) 
        b4statz_df.rename(columns={f"{VAR_COL_LISTING}.x": VAR_COL_LISTING}) 
        
        ## ii. apply harmonized as weighted on harmonized-level statistics
        df = do_normalize_a_dframe(b4stats_in=b4statz_df, df_in=df, is_intermediate_step=False) 
        return df 
        
    def get_infor(self, with_b4stats=False):
        '''TODO: 
            - rename func to match popular data zoo lingo, 
            - build formated report 
        '''
        # i. get info objects
        meta = [DatasourceMeta(cht) for cht in self.collection]
        a5statz = self.after_istatz 
        b4statz = self.b4_istatz if with_b4stats else None
        # ii. build formated report << TODO: 
        return (meta, a5statz, b4statz) if with_b4stats else (meta, a5statz) 
    
    def get_dataset(self, xtransform=None, ytransform=None, target_col=None):
        '''
        logic: requires that already generated; a unified_harmonized_csv exists in dstore
        - check if csv exists -- raise notgeneratedyet error if not 
        - pd read csv and filter to relevant fields only 
        - create DataframeDataset(pd) {item=image meta, image fpath, ylabel=is_normal, condition_label, skio} 
        - if with_meta, get stats object too 
        @return
        - DataFrameDataset object @ 50:50 norm/anom 
        '''
        assert target_col == None, f"Only one mode supported at the moment; mode curated_norm_anom"          ##TODO; language messages appropriately and consistently 
        if self.is_generated: 
            df = DataframeDataset(self.curated_csv, 
                                  xtransform=xtransform, 
                                  ytransform=ytransform, 
                                  target_col=target_col)            
            return df
        else:
            raise NotGeneratedYetError
            
    
    @property
    def is_generated(self):
        ''' True if unfied_harmnized-csv exists in dstore '''
        return self.curated_listing_exists 
    
    def __len__(self):
        return 0 if self.collection is None else len(self.collection) 
    
    

## Utils
### ========= Stats ====================
def get_image_stats(df_incoming, in_img_fpath_col, is_agg_only=False, drop_meta_colz=True, keep_listing=False):    
    statz_col_namez = ['min', 'max', 'median', 'mean', 'std', 'q1', 'q3',]                 
    channelz = ['red', 'green', 'blue'] 
    statz_headerz = statz_col_namez+[f"{chan}_{st}" for chan in channelz for st in statz_col_namez]
    meta_headerz = [VAR_COL_LISTING, VAR_COL_IMAGE_ID, 'is_file_valid', VAR_COL_IS_NORMAL]
    
    def tally_perc_normal(col):
        n = len(col)
        nnormz = np.sum(col==VALUE_IS_NORMAL)
        return nnormz/n 
    
    def do_quartile(col, q):
        return np.percentile(col, q) 
    
    median = lambda x: np.median(x) 
    q1 = lambda x: do_quartile(x, 25)
    q3 = lambda x: do_quartile(x, 75)
    
    agg_funcz = {'median': median, 
                 'q1' : q1, 
                 'q3' : q3, 
                 }
    
    tally_colz = [VAR_COL_IMAGE_ID, VAR_COL_IS_NORMAL] 
    pv_val_colz = tally_colz+statz_headerz
    
    def get_per_cohort_summary_statz(df, is_all=False):    
        ''' do within group statistics; even for harmonized b/c e.g. median is not commutative etc @ so don't just average cohortz to get to unified'''
        if is_all:
            df[VAR_COL_LISTING] = VALUE_ENTIRE_DATASET_LISTING_NAME   ## ENSURE order = call with cohorts before call @ entire dataset b.c this is scrambling src
            
        istatz = df.pivot_table(index=[VAR_COL_LISTING], 
                                    values=pv_val_colz,
                                    aggfunc={c: 'count' if c==VAR_COL_IMAGE_ID \
                                                            else tally_perc_normal if c==VAR_COL_IS_NORMAL \
                                                                else ('median' if 'median' in c \
                                                                    else q1 if 'q1' in c \
                                                                        else q3 if 'q3' in c \
                                                                            else 'mean') for c in pv_val_colz}
                                ).reset_index(
                                ).rename(
                                    columns={VAR_COL_IMAGE_ID: VAR_COL_STATZ_N_SAMPLES_ALL,
                                                VAR_COL_IS_NORMAL: VAR_COL_STATZ_PERC_SAMPLES_NORMAL} 
                                )
                                
        return istatz
    
    def per_image_stats(cht, fid, fp, nrm): 
        def unpacked_record(im):
            return [im.min(),
                    im.max(),
                    np.median(im),
                    im.mean(),
                    im.std(),
                    np.percentile(im, 25),
                    np.percentile(im, 75), ]
            
        # i. load it 
        fp = Path(fp)  
        fexists = fp.exists() and fp.is_file() 
        if fexists:
            im = skio.imread(fp) 
            chanz = im.shape[-1] if len(im.shape)==3 else 0
            rec = [ ]
            for c in [None]+list(range(chanz)):
                rec += unpacked_record( im if c is None else im[:, :, c] )
            im = None 
        else:
            rec = [np.nan]*len(statz_headerz) 
        return [cht, fid, fexists, nrm,] + rec 

    # i. do per image 
    df_statz = pd.DataFrame() 
    if is_agg_only: 
        pass
    else:
        # load  and calc per image VAR_COL_IMAGE_ID
        df_statz = [per_image_stats(cht, fid, fp, nrm) for _, (cht, fid, fp, nrm) in df_incoming[[VAR_COL_LISTING, VAR_COL_IMAGE_ID, in_img_fpath_col, VAR_COL_IS_NORMAL]].iterrows()]
        df_statz = pd.DataFrame.from_records(df_statz)
        df_statz.columns = meta_headerz + statz_headerz  

    # ii. get summary obj per cohort    
    istatz = get_per_cohort_summary_statz(df_statz)

    # iii. get summary of all & add to per cohort = unified 
    all_istatz = get_per_cohort_summary_statz(df_statz, is_all=True) 
    istatz = pd.concat([istatz, all_istatz]) 
    
    # iv. clean up 
    if drop_meta_colz:
        colz = statz_headerz+[VAR_COL_IMAGE_ID, 'is_file_valid',] 
        if keep_listing:
            colz += [VAR_COL_LISTING]
        df_statz = df_statz.filter([c for c in df_statz.columns if c in colz])
    
    return df_statz, istatz
