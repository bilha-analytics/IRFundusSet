---
title: 'IRFundusSet: A harmonized collection of public retinal fundus dataset'
tags:
    - python
    - ophthalmology
authors:
    - name: 
        given-names: P. Bilha 
        surname: Githinji
      orcid: 0009-0000-2080-4979 
      affiliation: "1"    
    - name: Keming Zhao 
      orcid: 0000-0002-0682-2780 
      affiliation: "2"
    - name: Jiantao Wang
      orcid:  0000-0000-0000-0000
      affiliation: "2"
    - name: Peiwu Qin
      orcid: 0000-0001-7336-7848
      corresponding: true
      affiliation: "1"
affiliations:
    - name: Tsinghua University, Tsinghua-Berkeley Shenzhen Institute, Shenzen, China
      index: 1
    - name: Shenzhen Eye Hospital, Jinan University, Shenzhen Eye Institute, Shenzhen, China
      index: 2
date: 17 October 2025
bibliography: paper.bib
---

# Summary
The Integrated Retinal Fundus Set (IRFundusSet) is a python  module that enables researchers to seamlessly integrate public Retinal Fundus Photograph (RFP) data sources into a single harmonized dataset for downstream machine learning or deep learning application. Integrating publicly available datasets offers access to a larger, more diverse, and more realistic data pool for research and development. Consuming disparate public datasets as a cohesive unit, however, is a non-trivial endevour due to heterogeneities such as varied data organization structures, equipment types and configurations, and definitions of targets such as what is a healthy or non-pathological observation. IRFundusSet consolidates and harmonizes ten public RFP data sources, encompassing 46,064 images, into a unified resource that aims to eliminate batch differences and offer a common definition for a non-pathological observation. Functionality includes automated metadata extraction, image pixel data standardization, and the creation of a curated label that identifies truly healthy eyes. Additionally, the module avails the unified dataset via an interface that streamlines integration with existing model training pipelines such as Pytorch.  


# Statement of need
The advancement and critical translation of artificial intelligence (AI) solutions for computational analysis and modeling of retinal fundus photographs (RFPs) necessitates access to larger-scale, diverse and representative datasets[@grzybowski_artificial_2023-2]. Public datasets are an invaluable resources for research, and while originally designed for specific research questions, collectively they can represent a wider problem space, offering a diverse and near real-world data pool. However, collective use of these datasets as a unified resource is encumbered by considerable fragmentation and a lack of standardization in design and dissemination. This lack of compatibillity across sources from different centers or time periods presents a non-trivial endeavour for researchers, requiring considerable effort on data prepration tasks that are not the primary focus of a research undertaking. For instance, crucial integration heterogeneties across these archives include varied directory organization structurees, inconsitent content metadata, disparities in raw image data characteristics, and ambiguous or conflicting definitions of essential target labels, particularly what constitutes a non-pathological observation[@khan_global_2021]. The Integrated Retinal Fundus Set (IRFundusSet) offloads the burden of consolidating disparate directory structures, establishing harmonization dimensions, preparing the standardization operations, and eventually establishing the unified dataset, allowing research efforts to focus on core scientific issues. 
 

# The data sources
IRFundusSet indexes a collection of 46,064 RFP images from 10 public archives, which are selected based on ease of access as well as for their potential to represent diverse properties for RFP modeling. Together, the sources capture multiple collection centers, several ethinicities and age groups, and common retinal pathologies like Diabetic Retinopathy (DR), Diabetic Macula Edema (DME), Age-relateed Macular Degeneration (AMD), Glaucoma, Cataracts and Pathological Myopia (PM). 
Table \autoref{tbl:cohortz} lists these sources and their properties. For brevity purposes, a detailed description of each source and its contribution is available in the associated pre-print for this record[@arxiv-entry]. 


**Table 1: Public retinal fundus datasets in IRFundusSet**
[][tbl:cohortz]
|  | n images | n src normal | \% left eye | \% curated | \% old was normal | \% new is normal |
|---|---|---|---|---|---|---|
| CHASEDB1 [@fraz_chase_db1_2012] | 28 | 0 | 0.50 | 1.00 | 0.000 | 0.000 |
| HRF [@odstrcilik_retinal_2013] | 45 | 15 | 0.47 | 1.00 | 0.333 | 0.289 |
| STARE [@hoover_locating_2000] | 397 | 36 | 0.46 | 1.00 | 0.091 | 0.076 |
| PAPILA [@kovalyk_papila:_2022] | 488 | 333 | 0.50 | 1.00 | 0.682 | 0.168 |
| IDRiD [@porwal_indian_2018] | 597 | 168 | 0.52 | 1.00 | 0.281 | 0.201 |
| Retina Cataracts [@noauthor_cataract_nodate] | 601 | 300 | 0.50 | 1.00 | 0.499 | 0.186 |
| FIVES [@jin_fives:_2022] | 800 | 200 | 0.47 | 1.00 | 0.250 | 0.176 |
| Kaggle1000 [@cen_automatic_2021] | 1000 | 38 | 0.42 | 1.00 | 0.038 | 0.038 |
| ODIR [@noauthor_odir-2019_nodate,@noauthor_ocular_nodate] | 7000 | 2816 | 0.50 | 1.00 | 0.402 | 0.117 |
| EyePACS [@noauthor_diabetic_nodate] | 35108 | 25802 | 0.50 | 0.41 | 0.735 | 0.149 |
| **Total** | 46064 | 29708 | 0.48 | 0.94 | 0.331 | 0.140 |
| **Total Without EyePACS** | 10956 | 3906 | 0.48 | 1.00 | 0.286 | 0.139 |




# Functionality
The Python modules are structured such that the user simply needs to download the source datasets to a local directory, and subsequently execute the Python package to parse, catalog and harmonize the retinal fundus images into a unified dataset. All or some of the identified sources can be considered for unification, and a template configuration (INI) file is provided, which specifies the location of included sources. Access points include a command line generate function and a dataset iterator (algined with PyTorch datasets) for seemless integration into modeling pipelines. A user guide in the form of a Jupyter Notebook~\cite{noauthor_jupyter_nodate} is included with the package.

```python
## Creating IRFundusSet Dataset object 
## Generates the unified dataset if it does not already exist
irf_dataset = IRFundusSet(out_dir="../output_irfundus_set__256",
                        ## Set output image sizes and harmonization method
                        out_img_w_size=256,
                        harmonize_method=None,
                        ## Set which of the 10 public sources to unify 
                        in_cohorts_config="../cohorts.ini", 
                        generate_only=False,
                        force_reload=False,
                        ## Setting which column to use for target label 
                        target_col=None,     
                        ## Provide transforms for X image features or y-target labels   
                        xtransform=None, 
                        ytransform=None,)
```


## Harmonizing the pixel data
The Python modules standardize both the meta information and pixel data of the retinal fundus images. Configurable options for meta properties include image size and output file format. Two statistical methods are available for harmonizing the pixel data, and application is first within the source-cohort level and then aggregated at the unified dataset level. The harmonization options are: 1. A standard method that makes use of the mean and standard deviation, and 2. A robust method, which employs the median and inter-quartile range.    


## Curating the not-pathological label
We leverage existing extensive literature (such as clinical literature, visual atlases and specialized guidelines) to resolve the varying definitions of what a healthy on non-pathological observation entails. Three rounds of manual curation determine a global non-pathological label, incrementally refining the quality of the label and eventually updating the consolidated data catalogue with this label.
Figure \autoref{fig:flow-chart}  summarizes the steps taken to arrive at this label.

![Flow chart depicting the process of curating non-pathological observations and creating the new `is_normal` label.\label{fig:flow-chart}](curate-flow.jpg){width="70%" .center-image}


## Summary of data properties
Of the 46,064 images, 25,406 images successfully undergo the curation and annotation process, resulting in 19,871 images being assigned a new `is_normal` label indicating if they are non-pathological or not. We determine 3,515 to be healthy/Normal/non-pathological across the sources. Figure \autoref{fig:img-propz}  qualititatively explores for biases in the newly curateed `is_normal` label using t-SNE components of the harmonized images (from standardizing with mean and standard deviation). There is no apparent clustering that discernably deliniates the images based on the label. Figure \autoref{fig:examples} is a snapshot of the resulting unified catalogue, illustrating the metadata collected for the images. 


![Properties of unified image data.\label{fig:img-propz}](image-xtics-plot1.jpg)

![Example image records in unified catalogue.\label{fig:examples}](sample-records-catalogue.jpg)


## Code availability
The Integrated Retinal Fundus Set (IRFundusSet) is publicly available on Github and Zenodo. The IRFundusSetPython modules are on Github at https://github.com/bilha-analytics/IRFundusSet, while the independent curated catalogue is on Zenodo at https://zenodo.org/records/10617824. 



# Acknowledgements
We thank the support from the National Natural Sci-ence Foundation of China 31970752; Science, Technology, Innovation Commission of Shenzhen Municipality JCYJ20190809180003689, JSGG20200225150707332,JCYJ20220530143014032,ZDSYS20200820165400003,WDZC20200820173710001, WDZC20200821150704001, JSGG20191129110812708,KCXFZ20211020163813019; Shenzhen Bay Laboratory Open Funding, SZBL2020090501004; Department of Chemical Engineering-iBHE special cooperation joint fund project, DCE-iBHE-2022-3; Tsinghua Shenzhen Interna-tional Graduate School Cross-disciplinary Research and Innovation Fund Research Plan, JC2022009; and 　 Bureau of Planning, Land and Resources of Shenzhen Municipality (2022) 207.
 
# References

