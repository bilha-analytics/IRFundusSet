'''
@bg
'''
import irfundusset.data_source as DS
    
def IRFundusSet(in_cohorts_config, 
                out_dir, 
                out_img_w_size, 
                force_regenerate=False, 
                clahe_b4_harmonize=False,
                xtransform=None, 
                ytransform=None, 
                target_col=None,
                method='zscore',
                generate_only=False, ):
    # i. generate  
    h = DS.HarmonizedDataSource(out_dir=out_dir,
                                  out_image_w=out_img_w_size,)    
    hstatus = h.generate(in_cohorts_config=in_cohorts_config, 
                   force_regenerate=force_regenerate,
                   method=method,
                   clahe_b4_harmonize=clahe_b4_harmonize,)  
       
    # ii. load and return dataset 
    return (hstatus, h.collection) if generate_only else h.get_dataset(xtransform=xtransform, 
                                                        ytransform=ytransform, 
                                                        target_col=target_col, )