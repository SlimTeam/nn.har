Comparison of Classification Methods for Human Activity Recognition
Teal Hobson-Lowther

Image and Multidimensional Signal Processing
EENG 510
12/14/2015

To see a description of the code contained in this zip, read the paper "HAR_Final.pdf". 

To run the code:

Naive Bayes Classification - open the file "NaiveBayes.m" contained in this folder. It requires the files
"PreProcessing.m" and "PUC_withUser.mat", both contained in the same folder, as well as the Statistics Toolkit,
sold separately from the basic Matlab package. Running "NaiveBayes.m" will perform
feature extraction on the data contained in "PUC_withUser.mat" and then generate a naive bayes classifer 
to classify this data. The performance of the classifier, on the testing data, will be contained in a
confusion matrix, C. 

Neural Network Classification - open the file "multilayernet.m" contained in this folder. It requires the
files "PreProcessing.m" and "PUC_withUser.mat", both contained in this same folder, as well as Matlab's
Neural Network Toolkit, sold separately from the basic Matlab package. Running "multilayernet.m" will perform
feature extraction on the data contained in "PUC_withUser.mat" and then generate a Neural Network structure
to classify this data. The performance of the classifier can be determined by looking at plots generated 
via the network training GUI that launches while this code is run. 

Enjoy! 
Teal


