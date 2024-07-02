package org.example;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.data.TrainTestFilesDataSet;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BeMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BiasedMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;

import java.io.IOException;
import java.util.Map;

public class Bemf {

    public Map<String, Double> bemfAgeGroup(String trainFile1, String trainFile18, String trainFile25, String trainFile35, String trainFile45, String trainFile50, String trainFile56, Map<String, Double> maeByAgeGroup, String testFile, DataModel datamodel) throws IOException {
        Double averageMAE1=findAverageMAE(trainFile1,testFile,datamodel,1);
        maeByAgeGroup.put("1", averageMAE1);
        //System.out.printf("Average MAE for age group 1 : "+totalMae/count);
        Double averageMAE18=findAverageMAE(trainFile18,testFile,datamodel,18);
        maeByAgeGroup.put("18", averageMAE18);

        Double averageMAE25=findAverageMAE(trainFile25,testFile,datamodel,25);
        maeByAgeGroup.put("25", averageMAE25);

        Double averageMAE35=findAverageMAE(trainFile35,testFile,datamodel,35);
        maeByAgeGroup.put("35", averageMAE35);

        Double averageMAE45=findAverageMAE(trainFile45,testFile,datamodel,45);
        maeByAgeGroup.put("45", averageMAE45);

        Double averageMAE50=findAverageMAE(trainFile50,testFile,datamodel,50);
        maeByAgeGroup.put("50", averageMAE50);

        Double averageMAE56=findAverageMAE(trainFile56,testFile,datamodel,56);
        maeByAgeGroup.put("56", averageMAE56);

        return maeByAgeGroup;
    }

    private static Double findAverageMAE(String trainFile,String testFile, DataModel datamodel,Integer agegroup) throws IOException {
        TrainTestFilesDataSet trainTestFilesDataSet = new TrainTestFilesDataSet(trainFile, testFile);
        DataModel datamodel1 = new DataModel(trainTestFilesDataSet);
        double[] ratings = {1.0, 2.0, 3.0, 4.0, 5.0};

        BeMF bemf = new BeMF(datamodel1,2,100,0.006,0.16,ratings);
        bemf.fit();


        double totalMae = 0.0;
        int count=0;
        for (TestUser testUser : datamodel1.getTestUsers()) {
            String userId1 = testUser.getId(); // User IDs start from 1
            String[] userData = datamodel.getDataBank().getStringArray(userId1);
            if (userData != null) {
                Integer ageInteger = Integer.parseInt(userData[0]);
                if (ageInteger.equals(agegroup)) {
                    double[] predictions = bemf.predict(testUser);
                    double mae = new MAE(bemf).getScore(testUser, predictions);
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
