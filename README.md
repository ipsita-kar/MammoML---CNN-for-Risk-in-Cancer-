# MammoML---CNN-for-Risk-in-Cancer-

![logo](https://github.com/ipsita-kar/MammoML---CNN-for-Risk-in-Cancer-/blob/main/POSTER.png)
<ul>
**  KEY FACTS**
  <li><strong>Breast cancer caused 670,000 deaths globally in 2022.</strong></li>
  <li><strong>Roughly half of all breast cancers occur in women with no specific risk factors other than sex and age.</strong></li>
  <li><strong>Breast cancer was the most common cancer in women in 157 countries out of 185 in 2022.</strong></li>
  <li><strong>Breast cancer occurs in every country in the world.</strong></li>
  <li><strong>Approximately 0.5–1% of breast cancers occur in men.</strong></li>
</ul>

![logo](https://github.com/ipsita-kar/MammoML---CNN-for-Risk-in-Cancer-/blob/main/FINAL.png)
_Overview_
Breast cancer is a disease in which abnormal breast cells grow out of control and form tumours. If left unchecked, the tumours can spread throughout the body and become fatal.

Breast cancer cells begin inside the milk ducts and/or the milk-producing lobules of the breast. The earliest form (in situ) is not life-threatening and can be detected in early stages. Cancer cells can spread into nearby breast tissue (invasion). This creates tumours that cause lumps or thickening. 

Invasive cancers can spread to nearby lymph nodes or other organs (metastasize). Metastasis can be life-threatening and fatal.
# MammoML: CNN for Cancer Screening
![logo](https://github.com/ipsita-kar/MammoML---CNN-for-Risk-in-Cancer-/blob/main/Raw%20Images.png)
## <h1>Overview</h1>
This project involves developing a Convolutional Neural Network (CNN) model using ResNet50 for breast cancer detection from mammograms. The model is built on the CBIS-DDSM dataset, an updated and standardized version of the Digital Database for Screening Mammography (DDSM). The goal is to facilitate the development and testing of decision support systems in mammography.
![logo](https://github.com/ipsita-kar/MammoML---CNN-for-Risk-in-Cancer-/blob/main/newplot.png)
## <h1>Mentor</h1>
**Professor Dr. Sundarakumar K B**  
**Dept. of Computer Science and Engineering**  
**Shiv Nadar University, Chennai**
</a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

## <h1>Project Details</h1>
**Duration:** October 2023 - November 2023  
**Model Used:** ResNet50  
**Programming Language:** Python  
**Libraries Used:** NumPy, Pandas, Matplotlib, Seaborn, TensorFlow  
![logo](https://github.com/ipsita-kar/MammoML---CNN-for-Risk-in-Cancer-/blob/main/CBIS%20DDSM%20DATASET.png)
## <h1>Dataset: CBIS-DDSM</h1>
The CBIS-DDSM (Curated Breast Imaging Subset of DDSM) dataset is a collection of medical images in JPEG format, derived from the original DDSM dataset, which was 163GB in size. The resolution of the images in the CBIS-DDSM dataset matches that of the original dataset. This dataset is focused primarily on breast imaging for mammography.
![logo](https://github.com/ipsita-kar/MammoML---CNN-for-Risk-in-Cancer-/blob/main/Data%20cleaning.png)
### <h1>Key Dataset Statistics:</h1>
**Number of Studies:** 6,775  
**Number of Series:** 6,775  
**Number of Participants:** 1,566  
**Number of Images:** 10,239  
**Modality:** MG (Mammography)  
**Image Size:** 6 GB in JPEG format  
![logo](https://github.com/ipsita-kar/MammoML---CNN-for-Risk-in-Cancer-/blob/main/Data%20cleaning.png)
### <h1>Dataset Description</h1>
The CBIS-DDSM dataset is a well-curated subset of the DDSM data, selected by a trained mammographer. The images have been decompressed and converted to DICOM format, and the dataset includes updated ROI (Region of Interest) segmentation and bounding boxes, along with pathologic diagnosis information for training data.
![logo](https://github.com/ipsita-kar/MammoML---CNN-for-Risk-in-Cancer-/blob/main/Data%20Visualizing.png)
### <h1>Challenges Addressed</h1>
Researchers in the field of mammography have faced challenges in replicating research results due to the lack of a standardized evaluation dataset. Most computer-aided diagnosis (CADx) and detection (CADe) algorithms for breast cancer in mammography rely on private datasets or unspecified subsets of public databases. The CBIS-DDSM dataset addresses these challenges by providing a well-curated, publicly accessible, and standardized version of the DDSM for future CAD research in mammography.

### <h1>Important Note</h1>
The dataset's structure assigns multiple patient IDs to each participant, which can be misleading. For instance, a participant may have 10 separate patient IDs, each containing information about different scans. Despite this, there are only 1,566 actual participants in the cohort.
**RESULTS **
![logo](https://github.com/ipsita-kar/MammoML---CNN-for-Risk-in-Cancer-/blob/main/RESULTTT.png)

**1)For calcification cancer, most cases are usually in the left breast.

2)Calcification cancer has 45 types, the majority of which are PLEOMORPHIC.**
## <h1>Model Performance</h1>
**Accuracy:** 96.8%  
**Results:** The model accurately predicted cancerous vs. non-cancerous mammograms on the test set.

