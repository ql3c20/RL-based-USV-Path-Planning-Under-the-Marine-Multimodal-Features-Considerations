# RL-based USV Path Planning Under the Marine Multimodal Features Considerations
* @Author: DavidLin
* @Date  : 2024/12/7
* @Contact : davidlin659562@gmail.com
* @Description : This code corresponds to the work "RL-based USV Path Planning Under the Marine Multimodal Features Considerations"
sumitted to IEEE Internet of Things Journal.


# The structure is as follows:
> Image processing module
* [Geographic Data] : Files that may be used in Image processing module.
* `step1_ECDIS.py`: Code to process ECDIS file.
* `step2_Discretization.py`: Code to rasterize the processed ECDIS file.
* `step3_Edge detection and Contour extraction.py`: Code to process random obstacles through Edge detection and Contour extraction.
> Meteorological analysis module
* [Geographic Data]: Files that may be used in Meteorological analysis module.
* `step1_nc_to_excel.py`: Code to process dataset file (.nc) to obtain excel.
* `step2_excel_to_vector field.py`: Code to process the meteorological data through Logarithmic formula transformation and Bilinear interpolation.
> Path planning module
* [Ablation Experiments]: files that may be used in Ablation Experiments.
* [Experiments for Comparison]: files that may be used in Experiments for Comparison.
* [Experiments for Testing Generalization Ability]: files that may be used in Experiments for Testing Generalization Ability.
* `fusion_DQN.py`: Code to complete reinforcement learning through fusion DQN proposed in the paper.
* `Multimodal_characteristics_Marine_environment.py`: Code to construct the the interactive environment for the agent to RL.


# Supplementary introduction
1. The preprocessing steps need to be completed according to the Image processing module and Meteorological analysis module.
(Of course, the processed data has also been prepared. )
2. The test data can ensure that all the experiments can be completed, which can be replaced by your own data.
3. In all files, the parts that can be adjustied have been marked, such as hyperparameters, file paths, environmental data, etc.
If you have any questions or suggestions, please feel free to contact me.

# Dataset download
To download the [Reanalysis CORAv1.0 dataset](https://mds.nmdis.org.cn/pages/dataViewDetail.html?dataSetId=83).
To download the [ERA5 dataset](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview).


# Contact email:
DavidLin: davidlin659562@gmail.com