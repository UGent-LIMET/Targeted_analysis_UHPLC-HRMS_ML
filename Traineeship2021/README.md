# Traineeship2021

## Figure representing the roadmap for the final scripts 

![Roadmap](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/Roadmap.png)


## The abstract of the project 

The emerging field of metabolomics is defined as the comprehensive measurement of all metabolites and low-molecular-weight molecules in a biological specimen. They can be detected with liquid chromatography-high-resolution mass spectrometry (LC-HRMS). One area within metabolomics comprises the targeted analysis of a subset of biomarkers that play an important role in a physiological condition of interest. 
To reduce time-intensive and manual work of the targeted data analysis, supervised machine learning (ML) models were created to predict the presence and abundance of metabolites in a sample. The main research question for this project is; can these machine learning models be a reliable tool to do these predictions out of the given data?
 
Some preparatory steps were performed on the data. The raw input data was converted and transformed into readable files. Out of these files, the target data was selected. For each metabolite standard per sample, two-dimensional contour charts of the region of interest were created. These images were used as train and test data together with the accompanying training labels. 
The ML assignment consisted of two parts;
One part was to create a classification model to predict if a metabolite is present in a sample, based on labelled examples. 
Part two was to create a regression model to predict the area of the metabolite, based on labelled examples, where the area was calculated on a defined chromatographic peak (traditional targeted approach). The area is linked to the abundance of the metabolite in a sample; and therefore a good approach to quantify the important biomarkers.
Multiple classification models were built. The best models were made with Random Forest (acc = 0.87), Gaussian Process (acc = 0.85) and convolutional neural network (acc = 0.73). Further improvements are expected by increasing the highly curated training dataset and by performing more hyperparameter tuning.
For the regression models, Random Forest regression was used (acc = 0.87). To increase this result, new images with a uniform scaling across all samples should be made. 
 
These results demonstrate the potential of supervised machine learning models to predict the presence and abundance of targeted metabolites in samples, but further improvements are needed.



## Figure representing the workflow of the project 

 ![pipeline_project_adapted](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/pipeline_project_adapted6.png)
