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
    """
    Create or load a harmonized IR fundus dataset.
    This function is a convenience wrapper around DS.HarmonizedDataSource that:
    1) constructs a HarmonizedDataSource configured to write/read data under out_dir
        with images resized to out_img_w_size,
    2) optionally runs the harmonization/generation step, and
    3) either returns the generation status and raw collection (generate_only=True)
        or returns the constructed dataset object (default).
    Parameters
    ----------
    in_cohorts_config : str or Mapping
            Path to or object describing the cohorts configuration used to locate and
            combine source datasets for harmonization.
    out_dir : str
            Output directory where harmonized data, cached artifacts and metadata are
            written/read.
    out_img_w_size : int
            Target width (and assumed height) for output images produced by the
            harmonization pipeline.
    force_regenerate : bool, optional (default=False)
            If True, force re-generation of harmonized outputs even if existing cached
            artifacts are present in out_dir.
    clahe_b4_harmonize : bool, optional (default=False)
            If True, apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            preprocessing to images before running harmonization. Useful to enhance
            local contrast on images from heterogeneous sources.
    xtransform : Callable or None, optional (default=None)
            Transformation or preprocessing applied to input images when building the
            final dataset (passed to h.get_dataset). Typical values are torchvision
            transforms or custom callables that accept an image and return a tensor.
    ytransform : Callable or None, optional (default=None)
            Transformation applied to labels/targets when building the final dataset.
    target_col : str or None, optional (default=None)
            Name of the column/field in the internal collection to use as the target
            variable. If None, the dataset builder will use its default target selection.
    method : str, optional (default='zscore')
            Harmonization method identifier passed to the harmonization pipeline (e.g.
            'zscore', or other methods supported by DS.HarmonizedDataSource.generate).
    generate_only : bool, optional (default=False)
            If True, only perform generation/harmonization and return the generation
            status and raw collection without building the dataset. If False, return
            the dataset object built from the harmonized collection.
    Returns
    -------
    tuple or object
            If generate_only is True, returns a tuple (hstatus, collection) where
            hstatus is the value returned by DS.HarmonizedDataSource.generate and
            collection is the harmonized collection object (internal metadata/manifest).
            If generate_only is False (default), returns the dataset object produced by
            h.get_dataset(...). The dataset type depends on the implementation of
            DS.HarmonizedDataSource (commonly a torch.utils.data.Dataset-like object).
    Notes
    -----
    - This function has side effects: it may create directories and write files under
        out_dir. Generation can be time- and resource-consuming depending on the
        size of input cohorts.
    - Any exceptions raised by DS.HarmonizedDataSource.generate or
        DS.HarmonizedDataSource.get_dataset are propagated to the caller.
    - The exact types and semantics of in_cohorts_config, hstatus, and the dataset
        depend on the DS.HarmonizedDataSource implementation in the surrounding codebase.
    Examples
    --------
    # Generate harmonized files and get the dataset (default behavior)
    dataset = IRFundusSet(config_path, "/tmp/out", 512)
    # Only run generation and inspect the returned collection
    status, collection = IRFundusSet(config_path, "/tmp/out", 512, generate_only=True)
    """
    
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