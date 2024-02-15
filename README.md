# NeuroOverParam
Repository for the codes used the Neuromorphic Overparameterisation paper

Description of codes and files:

**Feature Selection**
Feature_Selection.ipynb – This notebook will perform the feature selection algorithm for a specified file and task.
Feature_Selection_Analysis.ipynb - This notebook is used to analyse the results from the above notebook.
Feature_Selection_Figure_Plot.ipynb - This notebook plots the appropriate figures from the above analysis
The results for each task in the overparameterised and underparameterised states are saved as .pkl files which can be loaded directly in to the notebooks. 
.pkl file names are (task).pkl or (task)_under.pkl for over and underparameterised respectively

**Metrics**
Metrics.ipynb - This notebook will calculate the metrics, both for all channels with feature selection, and single channels without feature selection.
The plots are included in this notebook. 
metrics.pkl and metrics_interconnections.pkl contain the saved data from the metric calculations which can be loaded in to the notebook. 

**Overparameterisation**
Overparameterisation.ipynb - This notebook contains the code required to analyse the MSE as a function of number of parameters

**Meta-Learning**
Meta_Sines_Times.ipynb - This notebook contains the codes to run the Meta-Learning

**Benchmarks**
Benchmarks.ipynb - This notebook contains the codes necessary to run the benchmarking tasks. Specifically, one can define ESNs, MLPs and neural ODE's

**Data** 
Data folder contains all of the raw data. The data is seperated by task and network architecture. The excel spreadsheet has more details on which datasets form which architecture.
The folder is set up such that one can run the analysis files on a specific folder to evaluate the performance of a given network.
Data_results.zip - This contains the raw data and intermediate predictions and results for the feature selection and overparameterisation files. For some cases, one will need to download and extract this folder to reproduce the plots from the manuscript.

If there are any questions, issues or concerns with the code, please contact Kilian Stenning using k.stenning18@imperial.ac.uk
