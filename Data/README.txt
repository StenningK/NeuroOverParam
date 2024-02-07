Informatrion on how the data is organised
In the Data folder, there is all of the data every experimental run. The data is named as [Label].I-[input]_R-[Reservoir]
Reservoir corresponds to which sample is being measured - MS/PW/WM
Input refers to the input sequence. For labels 0,7 & 14 the input is 'MG' ie the original Mackey-Glass equation
For other labels, the input is the original reservoir followed by the index of the frequency channel selected. E.g. MS_300 means the 300th frequency channel from MS.
The arrays which say 'df_sum' means that the input is the sum of the channels specified
The arrays which say 'int_sum' means that the input is the sum of the channels specified after integrating the spectra
Those which say 'neg' have the FMR channel inverted and rescaled
For labels >39, the array is in the third layer and so the input contains information about both the first and second layer.   
E.g. MS_220_WM_220_ means we first measured MS and fed the 220th channel to WM, then we are taking the 220th channel from WM and using that as an input.

To ease data navigation we have split the data into the various networks explored

In Single folder, there is a folder for each sample with only the data from that sample

In Parallel folder, there is a folder for each parallel network

In Series folder, we have put each of the series networks in separate folders as described on the 'Networks' sheet

In PNN folder, there is one folder containing all of the data, and three folders which include the data from only one sample.

When running the analysis code, one can specify which folder to analyse. The code will load all data in that folder and analyse, saving the results to that folder.
