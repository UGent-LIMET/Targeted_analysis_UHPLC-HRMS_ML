# Electronic Traineeship Notebook - Introduction

*Traineeship 2021 :Development of a Machine Learning Strategy for Targeted Analysis of Metabolomics Data*

Loes Vervaele  // 19 april - 18 juni 2021



## Overview

1.  Introduction
2.  File Conversion
3.  Creating Images
4.  Getting test and train data
5.  Creating models
6.  Conclusion

# 1. Introduction

### 1.1 Introduction metabolites

The field of research at the LCA (Laboratorium voor Chemische Analyse) are metabolomics. Metabolomics is the analysis of the metabolites, those are substrates and products of the metabolism. This can be amino acids, nucleotides, lipids... and play are role in essential cellular functions like storage, signal transduction and energy production. These small molecules  (>1500 Da) can be detected in salvia, plasma, urine, tissue...  
Metabolites can be used as biomarkers to prognose a certain disease. For example plasma trimethylamine N-oxide (TMAO) is a marker for cardiovascular disease (CVD).

Detection happens via nuclear magnetic resonance or mass spectrometry. In addition a chromatography can be carried out.

For the analysis, multiple parameters are investigated; retention time, peak intensity, m/z. 

### 1.2 Description Pipeline 

The LCA uses a high-throughput pipeline to create an automatic data rapport of their samples. This pipeline is made in R. 
The raw input data derives from the device. Spectral data processing is performed so a datamatrix can be created. Some statistics (normalisation, transformation and scaling) are applied on the data.  Now this data can be used for different purposes; univariate analytics, unsupervised multivariate analytics and supervised multivariate analytics.

This project focusses on the univariate data-analytics.

### 1.3 Goal Project 

The goal of the project is to identify metabolites in samples, making use of predictions with a machine learning model. 

To use this model, different steps need to performed in advance. 

-   Transform raw data to a text or mzXML file

-   Make a dataframe out of the text/mzXML file

-   Combine the dataframe and targeted metadata to create an image

-   Use the images together with training labels to train and test a machine learning model (classification and regression)

    

    ![pipeline_project_adapted](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/pipeline_project_adapted6.png)

    

    *Image 1 : pipeline project*

    

    ![ML_reg](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/ML.png)

    

    *Image 2 : overview building machine learning model classification* 

    

    

    

    ![ML_reg](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/ML_reg2.png)

    
    
    *Image 3 : overview building machine learning model regression* 



# 2. File Conversion

## 2.1 Goal

The software of the measurement device (Thermo) delivers .raw files.  Now this conversion happens manually. The goal is to convert these files to .txt files with a script. This happens outside of the existing pipeline in an R-script running in Windows.

## 2.2 Example files

-   Example of a .raw file 

<img src="https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/19007sss1.png" style="zoom:75%;" />

*spectra 19007sss1 file*

<img src="https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/mz_spectra.png" style="zoom:75%;" />

*mz spectra from peak around 7 minutes RT*

-   example .txt file (200929s002.txt) 

    ```
    RunHeaderInfo
    dataset_id = 0, instrument_id = 0
    first_scan = 1, last_scan = 1272, start_time = 0.006873, end_time = 18.026191
    low_mass = 53.400002, high_mass = 800.000000, max_integ_intensity = 3918830800.000000, sample_volume = 0.000000
    sample_amount = 0.000000, injected_volume = 0.000000, vial = 0, inlet = 0
    err_flags = 0, sw_rev = 1
    Operator = 
    Acq Date = 
    comment1 = 
    comment2 = 
    acqui_file = 
    inst_desc = 
    sample volume units = 
    sample amount units = 
    Injected volume units = 
    Packet Position = 898194
    Spectrum Position = 55534702
    
    
    ScanHeader # 1
    position = 1, start_mass= 53.400002, end_mass = 800.000000
    start_time = 0.006873, end_time = 0.000000, packet_type = 21
    num_readings = 1089, integ_intens = 47863.262000, data packet pos = 0
    uScanCount = 0, PeakIntensity = 828.321900, PeakMass = 486.610422
    Scan Segment = 0, Scan Event = 0
    Precursor Mass  426.70 
    Collision Energy  0.00 
    Isolation width  746.60 
    Polarity positive, Profile Data, Full Scan Type, MS Scan
    SourceFragmentation Off, Type SingleValue, Values = 1, Mass Ranges = 0
    Turbo Scan Any, IonizationMode Electrospray, Corona Any
    Detector Any, Value = 0.00, ScanTypeIndex = -1
    DataPeaks
    
    Packet # 0, intensity = 0.000000, mass/position = 52.869401
    saturated = 0, fragmented = 0, merged = 0
    
    Packet # 1, intensity = 0.000000, mass/position = 52.869447
    saturated = 0, fragmented = 0, merged = 0
    
    Packet # 2, intensity = 0.000000, mass/position = 52.869492
    saturated = 0, fragmented = 0, merged = 0
    
    Packet # 3, intensity = 0.000000, mass/position = 52.869538
    saturated = 0, fragmented = 0, merged = 0
    
    Packet # 4, intensity = 0.000000, mass/position = 54.058997
    saturated = 0, fragmented = 0, merged = 0
    ...
    ```

    

-   example of .mzXML file (200929s002.mzXML)

    ```
    <?xml version="1.0" encoding="ISO-8859-1"?>
    <mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2"
           xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
           xsi:schemaLocation="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2 http://sashimi.sourceforge.net/schema_revision/mzXML_3.2/mzXML_idx_3.2.xsd">
      <msRun scanCount="1272" startTime="PT0.412351S" endTime="PT1081.57S">
        <parentFile fileName="file:////200929s002.raw"
                    fileType="RAWData"
                    fileSha1="6694899debd146394e9fb7cf3a5918679c564ae9"/>
        <msInstrument msInstrumentID="1">
          <msManufacturer category="msManufacturer" value="Thermo Scientific"/>
          <msModel category="msModel" value="Q Exactive"/>
          <msIonisation category="msIonisation" value="electrospray ionization"/>
          <msMassAnalyzer category="msMassAnalyzer" value="quadrupole"/>
          <msDetector category="msDetector" value="inductive detector"/>
          <software type="acquisition" name="Xcalibur" version="2.8-280502/2.8.1.2806"/>
        </msInstrument>
        <dataProcessing>
          <software type="conversion" name="ProteoWizard software" version="3.0.21124"/>
          <processingOperation name="Conversion to mzML"/>
        </dataProcessing>
        <scan num="1"
              scanType="Full"
              centroided="0"
              msLevel="1"
              peaksCount="1089"
              polarity="+"
              retentionTime="PT0.412351S"
              lowMz="52.869400991374"
              highMz="808.070333991576"
              basePeakMz="486.6104218"
              basePeakIntensity="828.3219"
              totIonCurrent="47863.262000000002"
              msInstrumentID="1">
          <peaks compressionType="none"
                 compressedLen="0"
                 precision="64"
                 byteOrder="network"
                 contentType="m/z-int"> ...(coded data)
          </peaks>
        </scan>
    ```

    

## 2.3 Code

Several options were tried before there was a running version of the code.

### 2.3.1 Version 1

The goal is to transform a .raw file to .txt file with an .exe from the firm. There is an example code available (see Git: REIMS_file_converter_microscript).
The script is adapted as followed : different input files are used and another executable (FileConverter.exe). The script is available on Git : Thermo_FileConverter.R. This script does not deliver the correct output. This is probably because the executable is protected by the firm.

### 2.3.2 Version 2

A second option is to use a program that is reversed engineered. This program was available online (https://en.freedownloadmanager.org/Windows-PC/Thermo-MSFileReader-FREE.html), but now it's no longer to be found. 

### 2.3.3 Version 3

A third option is to use MSConverter from ProteoWizard to convert the files to mzXML files. Also here, an example code was available (Git : File_conversion_example_code.R). This was adapted for the Thermo files, using msconvert.exe. There were some issues 

-    path to .exe file is not recognized as an internal or external command, operable program or batch file
-    execution failed with error code 1

The problem was that I didn't had admin rights to the folder where the executable was placed. When this was resolved, the script worked. 

The final code can be found below (Thermo_file_converter_microscript.R)

```R
# Title: Software framework for the processing and statistical analysis of multivariate MS-data
# Owner: Laboratory of Chemical Analysis (LCA), Ghent university
# Creator: Dr. Marilyn De Graeve, Loes Vervaele
# Running title: R pipeline
# Script: Part 0: Thermo file conversion using MSConvert (ProteoWizard)

# This script runs outside the R pipeline.
# Requirements: Windows OS and Thermo folder present on computer, no spaces in filenames, msconvert.exe installed
# To run, open in Rstudio, adjust the 'adjustments' below if needed and click "Source" to run the whole script
# No progressbars are shown, as long as red running symbol is visible, the script is calculating
# Check file conversion successful after finished (no symbols present)



########## Adjustments ##########

PATH_MSCONVERT <- 'C:/Windows/System32/MSConvert/msconvert.exe' # path to msconvert.exe
INPUT <- 'C:/Users/Loes/OneDrive - Hogeschool West-Vlaanderen/Vervaele Loes/0. Dashboard project/0. Testfiles/200929s002/' # path to .raw files
OUTPUT <- 'C:/Users/Loes/OneDrive - Hogeschool West-Vlaanderen/Vervaele Loes/0. Dashboard project/0. Testfiles/200929s002/' # path to the output folder
POLARITY <- 'negative'


#choose setting
OPTION1 <- 'pre-processing using R pipeline'
OPTION2 <- 'targeted analysis using ML'
setting <- OPTION2 # choose the wanted setting (OPTION1 or OPTION2)


# if OPTION1 : choose peak picking mode 
CENTROID <- 'peak picking in centroid mode'
PROFILE <- 'peak picking in profile mode'
PEAK_PICKING_MODE <- CENTROID 


########## 




# => DO NOT ADJUST CODE BELOW THIS POINT!
##########R Pipeline - Part 0: File_conversion##########
print(Sys.time())
start_time <- Sys.time()
print("R pipeline - Part 0: File_conversion - start!")
# Part I: File_conversion


#settings fixed
path_data_in <- INPUT
path_data_out <- OUTPUT


## setting working directory
setwd(path_data_in)


## converts raw file -> mzXML file
# selecting files 
filenames <- list.files(path_data_in, pattern="*.raw", recursive = TRUE) #exactive stalen
print(filenames)

# loop over the files
# selecting the correct settings 
for (x in filenames){
  if(setting == OPTION1){
    if(PEAK_PICKING_MODE == CENTROID){
      tryCatch({shell(paste0('"', '"', PATH_MSCONVERT, '" ', '"', '" --mzXML --64 --filter "peakPicking centroid msLevel=1-" --filter "polarity "', POLARITY ,'"', '" *.RAW -o "', path_data_out,'"'))})
    }
    if(PEAK_PICKING_MODE == PROFILE){
      tryCatch({shell(paste0('"', '"', PATH_MSCONVERT, '" ', '"', '" --mzXML --64 --filter "peakPicking cwt snr=0.1 peakSpace=0.1 msLevel=1-" --filter "polarity "', POLARITY ,'"', '" *.RAW -o "', path_data_out,'"'))})
    }
  }
  if(setting == OPTION2){
    tryCatch({shell(paste0('"', '"', PATH_MSCONVERT, '" ', '"', '" --mzXML --64 "', '" *.RAW -o "', path_data_out,'"'))})
  }
}

print("R pipeline - Part 0: File_conversion - done!")
print(Sys.time())
end_time <- Sys.time()
print(end_time - start_time)

#####################

```



# 3. Creating Images

## 3.1. Goal

The machine learning is not performed on the total dataset. This would be too much information. A region of interested is picked (based on the targeted metadata). Out of this, an image is created. The retention time is plotted on the x-axis, the m/z on the y-axis and the intensity on the z-axis . This 3D plot, is transformed to a 2D figure, using a coloured scalebar for the z-axis (see figure). 

![forming_images](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/forming_images.png)



## 3.2. Code

See Git : targeted_analysis_metabolomics_data_xml_to_png.py or targeted_analysis_metabolomics_data_txt_to_png.py, depending if the start file is a mz.XML file or .txt file. 

Both files are first transformed into a dataframe. Together with targeted metadata, the images are created.

![txt_to_image](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/txt_to_image.png)

*needed data from .txt file for dataframe*

![mzxml_to_image](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/mzxml_to_image.png)

*needed data from .mzXML file for dataframe*

![df_to_image](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/df_to_image.png)

*dataframe combined with targeted metadata for creating images*

Issues for regression : the scalebar is different, based on the maximum value of the concerning dataframe. This parameter needs to be normalized (for example log10) to create more uniform data. 



# 4 Getting test and train data

## 4.1 Goal

The goal is to create test and train data for machine learning purposes. Therefore, the png's (with correct name) and labels for train data are needed. 



## 4.2 Code

See Git : create_test_and_train_data.py



# 5 Creating Models

## 5.1 Goal

The goal is to perform supervised classification and regression on the data. 

Below, the models are divided according the purpose (classification/regression) and the amount of data used (because png's were created during the internship )



## 5.2 Classification Models  (150 images)

### 5.2.1 Random forest

-   Model 1 

    -   Model

        ```python
        PATH = "/media/sf_SF/Stage2021/test_ML/" #test directory
        #get data
        X_train,y_train = get_data_d(PATH+"ML_train/")
        X_test, y_test = get_data_d(PATH+"ML_test/")
        rf_classifier = RandomForestClassifier(max_depth = 2, random_state = 0)
        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)
        print(classification_report(y_test, y_pred))
        print("\nconfusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print("\naccuracy")
        print(accuracy_score(y_test, y_pred) * 100) 
        ```

        

    -   Result

        ```
                      precision    recall  f1-score   support
        
                   0       0.00      0.00      0.00         8
                   1       0.84      1.00      0.91        42
        
            accuracy                           0.84        50
           macro avg       0.42      0.50      0.46        50
        weighted avg       0.71      0.84      0.77        50
        
        
        confusion matrix
        [[ 0  8]
         [ 0 42]]
        
        accuracy
        84.0
        ```

        high accuracy but it is all classified with the same label => this is not good

    -   Too much data at once -> problem

    -   Let the model grow systematically for faster results

    -   Warm start (partial fit doesn't work here)

        

    


### 5.2.2 Gradient Boosting

-   Model 1 

    -   Model

        ```python
        PROJECT = "/media/sf_SF/Stage2021/test_ML/"
        
        X_train,y_train = get_data(PROJECT+"ML_train/")
        X_test, y_test = get_data(PROJECT+"ML_test/")
        
        clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        print("\nconfusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print("\naccuracy")
        print(accuracy_score(y_test, y_pred) * 100) 
        ```

        

    -   Result

        ```
                      precision    recall  f1-score   support
        
                   0       0.71      0.62      0.67         8
                   1       0.93      0.95      0.94        42
        
            accuracy                           0.90        50
           macro avg       0.82      0.79      0.80        50
        weighted avg       0.90      0.90      0.90        50
        
        
        confusion matrix
        [[ 5  3]
         [ 2 40]]
        
        accuracy
        90.0
        ```

-   Model 2

    -   Model

        ```python
        PROJECT = "/media/sf_SF/Stage2021/test_ML/"
        
        X_train,y_train = get_data(PROJECT+"ML_train/")
        X_test, y_test = get_data(PROJECT+"ML_test/")
        
        clf_gradientboost = GradientBoostingClassifier(n_estimators=150,learning_rate=0.8)
        clf_gradientboost.fit(X_train,y_train)
        
        
        y_pred = clf_gradientboost.predict(X_test)
        print(classification_report(y_test, y_pred))
        cf = confusion_matrix(y_test, y_pred)
        print(cf)
        print(accuracy_score(y_test, y_pred) * 100) 
        ```

        

    -   Result

        ```
                      precision    recall  f1-score   support
        
                   0       0.75      0.38      0.50         8
                   1       0.89      0.98      0.93        42
        
            accuracy                           0.88        50
           macro avg       0.82      0.68      0.72        50
        weighted avg       0.87      0.88      0.86        50
        
        [[ 3  5]
         [ 1 41]]
        88.0
        ```

        Not founds : more false negatives than true positives 

        

### 5.2.3 Bagging 

-   Model 1 

    -   Model

        ```python
        PROJECT = "/media/sf_SF/Stage2021/test_ML/"
        
        X_train,y_train = get_data(PROJECT+"ML_train/")
        X_test, y_test = get_data(PROJECT+"ML_test/")
        
        number_of_estimators = 100
        complexity = 10
        cart = LogisticRegression(C=complexity,solver='liblinear')
        
        lregbagging = BaggingClassifier(base_estimator=cart, n_estimators=number_of_estimators)
        lregbagging.fit(X_train,y_train)
        y_pred = lregbagging.predict(X_test)
        
        print(classification_report(y_test, y_pred))
        
        cf = confusion_matrix(y_test, y_pred)
        print(cf)
        print(accuracy_score(y_test, y_pred) * 100) 
        ```

        

    -   Result

        ```
                      precision    recall  f1-score   support
        
                   0       0.80      0.50      0.62         8
                   1       0.91      0.98      0.94        42
        
            accuracy                           0.90        50
           macro avg       0.86      0.74      0.78        50
        weighted avg       0.89      0.90      0.89        50
        
        [[ 4  4]
         [ 1 41]]
        90.0
        ```

        The not founds are 50% wrong labelled 

### 5.2.4 Boosting 

-   Model 1 

    -   Model

        ```python
        # Adaboost
        PROJECT = "/media/sf_SF/Stage2021/test_ML/"
        
        X_train,y_train = get_data(PROJECT+"ML_train/")
        X_test, y_test = get_data(PROJECT+"ML_test/")
        
        clf_adaboost = AdaBoostClassifier(n_estimators=150,learning_rate=0.9)
        clf_adaboost.fit(X_train,y_train)
        
        y_pred = clf_adaboost.predict(X_test)
        
        print(classification_report(y_test, y_pred))
        
        cf = confusion_matrix(y_test, y_pred)
        print(cf)
        print(accuracy_score(y_test, y_pred) * 100) 
        ```

        

    -   Result

        ```
                      precision    recall  f1-score   support
        
                   0       0.86      0.75      0.80         8
                   1       0.95      0.98      0.96        42
        
            accuracy                           0.94        50
           macro avg       0.91      0.86      0.88        50
        weighted avg       0.94      0.94      0.94        50
        
        [[ 6  2]
         [ 1 41]]
        94.0
        ```

        Best model so far 

### 5.2.5 Logistic Regression 

-   Model 1 

    -   Model

        ```python
        # Adaboost met logistic regression classifier
        PROJECT = "/media/sf_SF/Stage2021/test_ML/"
        
        X_train,y_train = get_data(PROJECT+"ML_train/")
        X_test, y_test = get_data(PROJECT+"ML_test/")
        
        cart = LogisticRegression(C=complexity,solver='liblinear')
        logreg_adaboost = AdaBoostClassifier(base_estimator=cart,n_estimators=150,learning_rate=0.9) 
        logreg_adaboost.fit(X_train,y_train)
        
        y_pred = logreg_adaboost.predict(X_test)
        
        print(classification_report(y_test, y_pred))
        
        cf = confusion_matrix(y_test, y_pred)
        print(cf)
        print(accuracy_score(y_test, y_pred) * 100) 
        ```

        

    -   Result

        ```
                      precision    recall  f1-score   support
        
                   0       0.80      0.50      0.62         8
                   1       0.91      0.98      0.94        42
        
            accuracy                           0.90        50
           macro avg       0.86      0.74      0.78        50
        weighted avg       0.89      0.90      0.89        50
        
        [[ 4  4]
         [ 1 41]]
        90.0
        ```

        The not founds are 50% wrong labelled 



## 5.3 Classification Models  (5110 images)

### 5.3.1 Testing multiple models at once

```python
names = ["Nearest_Neighbors", "Linear_SVM", "Polynomial_SVM", "RBF_SVM", "Gaussian_Process",
         "Gradient_Boosting", "Decision_Tree", "Extra_Trees", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "QDA", "SGD"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="poly", degree=3, C=0.025),
    SVC(kernel="rbf", C=1, gamma=2),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0),
    DecisionTreeClassifier(max_depth=5),
    ExtraTreesClassifier(n_estimators=10, min_samples_split=2),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(n_estimators=100),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier(loss="hinge", penalty="l2")]
    
scores = []
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores.append(score)

df = pd.DataFrame()
df['name'] = names
df['score'] = scores
df

sns.set(style="whitegrid")
ax = sns.barplot(y="name", x="score", data=df)
```

<img src="https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/multiple_models_scores.png" style="zoom:70%;" />

<img src="https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/multiple_models_plot.png" alt="multiple_models_plot" style="zoom:67%;" />



### 5.3.2 Random Forest

-   model 1

    -   model 	

        ```python
        # Random Forest Classifier
        X_train,y_train = load_X_if_matched_in_y(filenames_X_train, y)
        X_test, y_test = load_X_if_matched_in_y(filenames_X_test, y)
        
        number_of_trees = 1000
        max_number_of_features = 2
        
        #Make model
        RFCmodel = RandomForestClassifier(n_estimators=number_of_trees, max_features=max_number_of_features)
        
        #Train model
        RFCmodel.fit(X_train,y_train)
        
        #Test model
        y_pred = RFCmodel.predict(X_test)
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print("\naccuracy")
        print(accuracy_score(y_test, y_pred) * 100) 
        ```

        ​	

    -   Result 

        ```
                      precision    recall  f1-score   support
        
               FOUND       0.86      0.95      0.90       770
           NOT_FOUND       0.87      0.71      0.78       401
        
            accuracy                           0.86      1171
           macro avg       0.87      0.83      0.84      1171
        weighted avg       0.86      0.86      0.86      1171
        
        [[728  42]
         [117 284]]
        
        accuracy
        86.42186165670367
        ```


-   model 2

    -   model (with saving model + write filename with probabilities to file) 

        ```python
        # Random Forest Classifier
        X_train,y_train = load_X_if_matched_in_y(filenames_X_train, y)
        X_test, y_test = load_X_if_matched_in_y(filenames_X_test, y)
        
        number_of_trees = 5000
        max_number_of_features = 2
        
        #Maken van het model
        RFCmodel = RandomForestClassifier(n_estimators=500, max_depth=150, max_features=max_number_of_features)
        
        #Trainen van het model
        RFCmodel.fit(X_train,y_train)
        
        filename = 'test_model.sav'
        pickle.dump(RFCmodel, open(filename, 'wb'))
         
        ##############################################################################
        #opening model in different script
        
        import pickle
        
        # Load model 
        loaded_model = pickle.load(open('test_model.sav', 'rb'))
        result = loaded_model.score(X_test, y_test)
        print(result)
        
        # Test Random Forest Classifier
        y_pred = loaded_model.predict(X_test)
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print("\naccuracy")
        print(accuracy_score(y_test, y_pred) * 100) 
        
        # Test Random Forest Classifier
        y_pred_proba = loaded_model.predict_proba(X_test)
        print(y_pred_proba)
        ```

        

    -   result

        ```
                      precision    recall  f1-score   support
        
                   0       0.84      0.71      0.77       392
                   1       0.86      0.93      0.89       758
        
            accuracy                           0.85      1150
           macro avg       0.85      0.82      0.83      1150
        weighted avg       0.85      0.85      0.85      1150
        
        [[278 114]
         [ 53 705]]
        
        accuracy
        85.47826086956522
        ```

        

    -   probabilities

        ```
        [[0.1502     0.8498    ]
         [0.1708     0.8292    ]
         [0.7752     0.2248    ]
         ...
         [0.12666667 0.87333333]
         [0.4426     0.5574    ]
         [0.05413333 0.94586667]]
        ```

        

### 5.3.3 SVM

-   model 1

    -   ```python
        from sklearn import datasets, svm, metrics
        from sklearn.metrics import accuracy_score
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score
        from sklearn.neural_network import MLPClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix, accuracy_score
        
        svm_classifier = svm.SVC(gamma=0.001, kernel='rbf')
        svm_classifier.fit(X_train, y_train)
        
        predicted = svm_classifier.predict(X_test)
        print("predicted = {}".format(predicted))
        print("test set  = {}".format(y_test))
        acc = accuracy_score(y_test, predicted)
        print("accuracy  = {}".format(acc))
        
        cm = confusion_matrix(y_test, predicted)
        print(cm)
        
        
        ```

        

```
predicted = [1 1 1 ... 1 1 1]
test set  = [1 0 1 ... 1 1 1]
accuracy  = 0.6962774957698815

[[ 67 343]
 [ 16 756]]

```



### 5.3.4 Boosting 

-   model 1

    -   model

        same as 5.2.4 

    -   result 

        ```
                      precision    recall  f1-score   support
        
               FOUND       0.86      0.90      0.88       770
           NOT_FOUND       0.79      0.73      0.76       401
        
            accuracy                           0.84      1171
           macro avg       0.83      0.81      0.82      1171
        weighted avg       0.84      0.84      0.84      1171
        
        [[694  76]
         [110 291]]
         
        accuracy
        84.11614005123825
        
        ```

        



### 5.3.5 CNN

-   Model 1 

    -   Model

        ```python
        # Reshape
        X_train = X_train.reshape((len(X_train),128, 128,1))
        X_test = X_test.reshape((len(X_test),128, 128,1)) 
        
        # Normalisation
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
            X_train /= 255
            X_test /= 255
        
        # Model with parameters 
        
        batch_size = 32 # 
        epochs = 20 # 
        
        num_classes = 10
        img_rows, img_cols = 128, 128
        input_shape = (img_rows, img_cols,1)
        
        # Model
        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape)) 
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Dropout(0.3)) # Value between 0 and 1 
        
        model.add(BatchNormalization())
        
        model.add(Conv2D(32, (3, 3), activation='relu')) 
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Dropout(0.3)) # Value between 0 and 1 
        
        model.add(BatchNormalization())
        
        model.add(Flatten()) 
        model.add(Dense(50, activation='relu')) 
        
        model.add(Dropout(0.2)) # Value between 0 and 1 
        
        model.add(Dense(num_classes, activation='softmax')) # relu nog eens testen
        
        
        # 1
        model.compile(loss="categorical_crossentropy",
                      optimizer='adam',
                      metrics=['accuracy'])
        # 2
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        
        history = model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_test, y_test))
        ```


        
    
    -   Result
    
        ```
        accuracy score: 73.90180878552972


​        
                      precision    recall  f1-score   support
        
                   0       0.88      0.28      0.43       400
                   1       0.72      0.98      0.83       761
        
            accuracy                           0.74      1161
           macro avg       0.80      0.63      0.63      1161
        weighted avg       0.77      0.74      0.69      1161
        
        [[113 287]
         [ 16 745]]
        ```


​        

![](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/accuracy_loss_1.png)



-   Model 2

    ```python
    # Neural network parameters
    
    batch_size = 32 # 
    epochs = 150 # 
    
    num_classes = 10
    img_rows, img_cols = 128, 128
    input_shape = (img_rows, img_cols,1)
    
    # Model
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape)) 
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Dropout(0.3)) # Value between 0 and 1 
    
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, (3, 3), activation='relu')) 
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Dropout(0.3)) # Value between 0 and 1 
    
    model.add(BatchNormalization())
    
    model.add(Flatten()) 
    model.add(Dense(50, activation='relu')) 
    
    model.add(Dropout(0.2)) # Value between 0 and 1 
    
    model.add(Dense(num_classes, activation='softmax')) # relu nog eens testen
    
    
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))
    
    ```

    ![](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/accuracy_loss_2.png)

```
... 

Epoch 130/150
74/74 [==============================] - 81s 1s/step - loss: 0.1954 - accuracy: 0.9159 - val_loss: 0.5104 - val_accuracy: 0.8392
Epoch 131/150
74/74 [==============================] - 69s 925ms/step - loss: 0.2018 - accuracy: 0.9123 - val_loss: 0.4707 - val_accuracy: 0.8358
Epoch 132/150
74/74 [==============================] - 76s 1s/step - loss: 0.2007 - accuracy: 0.9235 - val_loss: 8.3036 - val_accuracy: 0.6690
Epoch 133/150
74/74 [==============================] - 80s 1s/step - loss: 0.1736 - accuracy: 0.9205 - val_loss: 0.4247 - val_accuracy: 0.8246
Epoch 134/150
...
```

-   model 3 : with dimensions image : 64 instead of 128  = error 

    ```python
    import tensorflow as tf
    
    from tensorflow.keras import datasets, layers, models
    import matplotlib.pyplot as plt
    
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=10, 
                        validation_data=(X_test, y_test))
    ```


    => error 
    
    ```
        ValueError: Input 0 of layer sequential is incompatible with the layer: : expected min_ndim=4, found ndim=2. Full shape received: (None, 16384)
    
    ```

-   Model 4 (1e Conv2D : kernel_size = 9x9)

    

    -   ```python
        # Neural network parameters
        
        batch_size = 32 # 
        epochs = 70 # 
        
        num_classes = 2
        img_rows, img_cols = 128, 128
        input_shape = (img_rows, img_cols,1)
        
        # Model
        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=(9, 9), activation='relu',input_shape=input_shape)) 
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Dropout(0.3)) # Value between 0 and 1 
        
        model.add(BatchNormalization())
        
        model.add(Conv2D(32, (3, 3), activation='relu')) 
        
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Dropout(0.3)) # Value between 0 and 1 
        
        model.add(BatchNormalization())
        
        model.add(Flatten()) 
        model.add(Dense(50, activation='relu')) 
        
        model.add(Dropout(0.2)) # Value between 0 and 1 
        
        model.add(Dense(num_classes, activation='relu')) # relu / softmax
        
        
        
        ```

    -   ```
        accuracy score: 35.153583617747444
        
        
                      precision    recall  f1-score   support
        
                   0       0.35      1.00      0.52       412
                   1       0.00      0.00      0.00       760
        
            accuracy                           0.35      1172
           macro avg       0.18      0.50      0.26      1172
        weighted avg       0.12      0.35      0.18      1172
        
        [[412   0]
         [760   0]]
         
         => all labeled as found ! 
        ```

        ![accuracy_4](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/accuracy_4.png)

-   Model 5

    -   ```python
        # Reshape
        X_train = X_train.reshape((len(X_train),128, 128,1))
        X_test = X_test.reshape((len(X_test),128, 128,1)) 
        
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))
        
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        history = model.fit(X_train, y_train, epochs=70, 
                            validation_data=(X_test, y_test))
                            
        # Performantie op de test data
        from sklearn.metrics import confusion_matrix, accuracy_score
        from sklearn.metrics import classification_report
        y_pred = model.predict_classes(X_test)
        print('\n')
        print('accuracy score:', accuracy_score(y_test, y_pred) * 100) 
        print('\n')
        print(classification_report(y_test, y_pred))
        cf = confusion_matrix(y_test, y_pred)
        print(cf)
        ```

        

    -   ```
        accuracy score: 66.10312764158918
        
        
                      precision    recall  f1-score   support
        
                   0       0.00      0.00      0.00       401
                   1       0.66      1.00      0.80       782
        
            accuracy                           0.66      1183
           macro avg       0.33      0.50      0.40      1183
        weighted avg       0.44      0.66      0.53      1183
        
        [[  0 401]
         [  0 782]]
        ```


### 5.3.6 Gaussian Process

-   model 1

    -   model

        ```python
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.metrics import confusion_matrix, accuracy_score
        from sklearn.metrics import classification_report
        from sklearn.gaussian_process.kernels import RBF
        
        clf_GP = GaussianProcessClassifier(1.0 * RBF(1.0))
        
        clf_GP.fit(X_train, y_train)
        score = clf_GP.score(X_test, y_test)
        
        y_pred = clf_GP.predict(X_test)
        
        print('accuracy score:', accuracy_score(y_test, y_pred) * 100) 
        print('\n')
        print(classification_report(y_test, y_pred))
        cf = confusion_matrix(y_test, y_pred)
        print(cf)
        ```

        

    -   result

        ```
        accuracy score: 85.08845829823083
        
        
                      precision    recall  f1-score   support
        
                   0       0.78      0.83      0.80       436
                   1       0.90      0.86      0.88       751
        
            accuracy                           0.85      1187
           macro avg       0.84      0.85      0.84      1187
        weighted avg       0.85      0.85      0.85      1187
        
        [[361  75]
         [102 649]]
        ```

        

        

## 5.4. Regression (7377 images)

### 5.4.1 Random Forest Regressor 

-   Model 1 

    -   code : test_regression_7337_RF
    -   result

    ```
    Fitting Random Forest regression on training set
    
    
    Getting Model Accuracy...
    Training Accuracy =  0.9686273874957818
    Test Accuracy =  0.7719067833913993
    ```

    ![](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/Regression_scatterplot.png)

    ​	=> all not founds are predicted wrong  => split the founds and not founds 

-   model 2 : the founds and not founds are split up 

    -   code : test_regression_found_nf_separated 

    -   result 

        -   Found

            ```
            Fitting Random Forest regression on training set
            
            
            Getting Model Accuracy...
            Training Accuracy =  0.9628369093819058
            Test Accuracy =  0.8406057837600825
            
            ```

            Via excel : manually R^2 calculated (=test accuracy) to check 

            ![calculation_R2_RF_7377](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/calculation_R2_RF_7377.png)

            Different R^2 with plotting the linear regression with Excel (due to other calculation algoritms)

        -   Not found 

            ```
            Fitting Random Forest regression on training set
            
            
            Getting Model Accuracy...
            Training Accuracy =  1.0
            Test Accuracy =  1.0
            
            ```

    To do : adapt the png's so they have a uniform scalebar

    The same maximum (intensity) for the scalebar delivers a lot of blanco png's. 
    That is why a normalization of the data is required before going further with the regression. 

# 6 Conclusion



This figure shows the roadmap for using the final scripts : 

![Roadmap](https://github.com/LoesVervaele/Traineeship2021/blob/main/ETN/images/Roadmap.png)



The best model for classification was with Random forest (acc = 0.87). The best model for regression is with also Random forest (acc = 0.87).

These results demonstrate the potential of supervised machine learning models to predict the presence and abundance of targeted metabolites in samples, but further improvements are needed.
