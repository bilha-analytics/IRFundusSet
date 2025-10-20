# IRFundusSet
Ocular conditions are a global concern and computational tools utilizing retinal fundus color photographs can aid in routine screening and management. Obtaining comprehensive and sufficiently sized datasets, however, is non-trivial for the intricate retinal fundus, which exhibits heterogeneities within pathologies, in addition to variations from demographics and acquisition. Moreover, retinal fundus datasets in the public space suffer fragmentation in the organization of data and definition of a healthy observation. We present Integrated Retinal Fundus Set (IRFundusSet), a dataset that consolidates, harmonizes and curates several public datasets, facilitating their consumption as a unified whole and with a consistent is_normal label. IRFundusSet comprises a Python package that automates harmonization and avails a dataset object in line with the PyTorch approach. Moreover, images are physically reviewed and a new is_normal label is annotated for a consistent definition of a healthy observation. Ten public datasets are initially considered with a total of 46064 images, of which 25406 are curated for a new is_normal label and 3515 are deemed healthy across the sources.

## Curated catalogue on Zenodo
The curated catalogue is available [on Zenodo here](https://zenodo.org/doi/10.5281/zenodo.10617823)


# Usage

**Installation**

```pip install irfundusset``` 


**Data Sources**

Identified datasets are in the public space and, therefore, we leave it to the researcher to access and directly download the datasets. 

For testing purposes, you can download [CHASEDB](https://www5.informatik.uni-erlangen.de/fileadmin/research/datasets/fundus-images/all.zip), which is currently the smallest source in the collection (about 73 Mb). 


**Generate integrated dataset**

Once downloaded, obtain the template configuration file `template_set_cohorts.ini` and indicate where the datasets are located. 

Then initialize your IRFundusSet data object as shown below to generated the consolidated dataset and save it to your desired location.

```python
## Creating IRFundusSet Dataset object 
## Generates the unified dataset if it does not already exist
irf_dataset = IRFundusSet(out_dir="../output_irfundus_set__256",
                        ## Set output image sizes and harmonization method
                        out_img_w_size=256,
                        harmonize_method=None,
                        clahe_b4_harmonize=False,
                        ## Set which of the 10 public sources to unify 
                        in_cohorts_config="../cohorts.ini", 
                        generate_only=False,
                        force_regenerate=False, 
                        ## Setting which column to use for target label 
                        target_col=None,     
                        ## Provide transforms for X image features or y-target labels   
                        xtransform=None, 
                        ytransform=None,)
```



**User guide (Jupyter Notebook)**
An introductory user guide in the form of Jupyter Notebook is available at the root of this repository. The file name is `irfundusset_user_guide.ipynb`


# License

[MIT License](License.txt)

# Citing 
```bibtex
@misc{githinji2024irfundussetintegratedretinalfundus,
      title={IRFundusSet: An Integrated Retinal Fundus Dataset with a Harmonized Healthy Label}, 
      author={P. Bilha Githinji and Keming Zhao and Jiantao Wang and Peiwu Qin},
      year={2024},
      eprint={2402.11488},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2402.11488}, 
}
```

**Citing only the catalogue**
```
Githinji, P. B., Zhao, K., Wang, J., & Qin, P. (2024). Curated catalogue for IRFundusSet (Integrated Retinal Fundus Set) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10819965
```

