In this branch the performance of the recommendations for each gender group is evaluated on the balanced and imbalanced datasets. Models are also seperately trained according to gender on both balanced and imbalanced data. Prediction error is computed using MAE.

The dataset is balanced by deleting random male users until there are equal number of male and female users. The dataset balanced by gender is saved as balancedbygendergroup.csv in Gender data.zip. 
Both balanced and imbalanced datasets are divided to into datasets with only male and female users to train the model seperately these datasets are also given in Gender data.zip
abc-test.csv in Gender data.zip is the test file.

The code uses java version 21

Inorder to run the code, clone the branch, upload the folder and run the program in any IDE that supports java. 
Use balancedbygendergroup.csv and abc-test.csv and update the path to the file in line 19 and 20 respectively in Main.java as per the location of file in your local.
Use users_groupM.csv, users_groupF.csv and abc-test.csv and update the path to the file in line 12, 13 and 16 respectively in SeperatelyTrainedGenderGroup.java.
Use balacedbygendermale.csv, balacedbygenderfemale.csv and abc-test.csv and update the path to the file in line 11, 12 and 15 respectively in Septrainedbalanced.java.
