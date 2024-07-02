package org.example;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.data.TrainTestFilesDataSet;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.knn.UserKNN;
import es.upm.etsisi.cf4j.recommender.knn.userSimilarityMetric.UserSimilarityMetric;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;

import java.io.IOException;
import java.util.Map;

public class Knn {

    public Map<String, Double> knnAgeGroup(String trainFile1, String trainFile18, String trainFile25, String trainFile35, String trainFile45, String trainFile50, String trainFile56, Map<String, Double> maeByAgeGroup, String testFile, DataModel datamodel, UserSimilarityMetric metric) throws IOException {
        Double averageMAE1=findAverageMAE(trainFile1,testFile,datamodel,1,metric);
        maeByAgeGroup.put("1", averageMAE1);
        //System.out.printf("Average MAE for age group 1 : "+totalMae/count);
        Double averageMAE18=findAverageMAE(trainFile18,testFile,datamodel,18,metric);
        maeByAgeGroup.put("18", averageMAE18);

        Double averageMAE25=findAverageMAE(trainFile25,testFile,datamodel,25,metric);
        maeByAgeGroup.put("25", averageMAE25);

        Double averageMAE35=findAverageMAE(trainFile35,testFile,datamodel,35,metric);
        maeByAgeGroup.put("35", averageMAE35);

        Double averageMAE45=findAverageMAE(trainFile45,testFile,datamodel,45,metric);
        maeByAgeGroup.put("45", averageMAE45);

        Double averageMAE50=findAverageMAE(trainFile50,testFile,datamodel,50,metric);
        maeByAgeGroup.put("50", averageMAE50);

        Double averageMAE56=findAverageMAE(trainFile56,testFile,datamodel,56,metric);
        maeByAgeGroup.put("56", averageMAE56);

        return maeByAgeGroup;
    }

    private static Double findAverageMAE(String trainFile,String testFile, DataModel datamodel,Integer agegroup,UserSimilarityMetric metric) throws IOException {
        TrainTestFilesDataSet trainTestFilesDataSet = new TrainTestFilesDataSet(trainFile, testFile);
        DataModel datamodel1 = new DataModel(trainTestFilesDataSet);


        UserKNN userKNN;
        userKNN = new UserKNN(datamodel1, 75,metric, UserKNN.AggregationApproach.WEIGHTED_MEAN);
        userKNN.fit();


        double totalMae = 0.0;
        int count=0;
        for (TestUser testUser : datamodel1.getTestUsers()) {
            String userId1 = testUser.getId(); // User IDs start from 1
            String[] userData = datamodel.getDataBank().getStringArray(userId1);
            if (userData != null) {
                Integer ageInteger = Integer.parseInt(userData[0]);
                if (ageInteger.equals(agegroup)) {
                    double[] predictions = userKNN.predict(testUser);
                    double mae = new MAE(userKNN).getScore(testUser, predictions);
                    //System.out.println(mae);
                    if (!Double.isNaN(mae)) {
                        totalMae = totalMae + mae;
                        count++;
                    }
                } else {
                    continue;
                }
            }
        }
        return totalMae/count;


    }

}
