In this branch the performance of the recommendations for each age group is evaluated on the dataset balanced by age groups.
Prediction error is computed using MAE.

The dataset is balanced by deleting random users from each age group until there are equal number of users in all groups. 
The dataset balanced by age group is saved as balancedbyagegroup.csv in balanced age group data.zip
abc-test.csv in balanced age group data.zip is the test file. 

The code uses java version 21 Inorder to run the code, clone the branch, upload the folder and run the program in any IDE that supports java. 
Use the users.csv file in the balanced age group data.zip and update the path to the file in line 16 in Main.java as per the location of file in your local.
Use balancedbyagegroup.csv and abc-test.csv and update the path to the file in line 19 and 20 respectively in Main.java as per the location of file in your local.
