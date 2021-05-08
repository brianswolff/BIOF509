# BIOF509 Final Project

The goal of this project is to examine what metabolites might best relate to cancer and cancer-related fatigue. This project uses two machine learning algorighthms, a random forest classifier and linear support vector machine. They are used to classify patients into "healthy" or "cancer" classes, and also classify cancer patients as "fatigued" or "not fatigued", and then determine which metabolites are most important for those classifications.

## Data Description

There are three data files. One is "metabolomicsData.csv", which has all the measured metabolite concentrations from patient blood samples. The second is "metaboliteInfo.csv", which has information about each metabolite. These two files have the same row indexes corresponding to each metabolite, so they can be easiliy joined on their indexes. The third file is "patientData.csv", which has the rest of the data from patients, including demographic inforamtion, questionnaires, and behavior data. There is missing data here, in particular all demographic/behavior data from the healthy controls. But there is enough data for this project. Each row is a patient identifier, which is the same identifier used as column names in "metabolomicsData.csv".

## Running the code

There is only one script, "BIOF509_final.py", with the data pipeline contained in the "MetabolomicsML" class. At the end of the file, the full pipeline is implemented using both of the ML algorithms and several labels selected from the patient data; rank-ordered importance of the features (metabolites) is then written into a "results" directory. Other labels from the patient data can be easily chosen by entering the appropriate "patientData.csv" column name into the "classifier" method.
