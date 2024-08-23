package org.example;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BNMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BeMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BiasedMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class BeMf {

    public void runBemf(DataModel datamodel, DataModel dataModel1) throws IOException {
        BeMF beMf;
        double[] ratings = {1.0, 2.0, 3.0, 4.0, 5.0};

        beMf = new BeMF(dataModel1,2,100,0.006,0.16,ratings);
        beMf.fit();

        Map<String, Double> maeByGender = new HashMap<>();
        maeByGender.put("M", 0.0);
        maeByGender.put("F", 0.0);

        // Map to store counts for each gender
        Map<String, Integer> countByGender = new HashMap<>();
        countByGender.put("M", 0);
        countByGender.put("F", 0);

        for (TestUser testUser : dataModel1.getTestUsers()) {
            String userId = testUser.getId();
            String[] userInfo = datamodel.getDataBank().getStringArray(userId);
            if (userInfo != null) {
                String gender = userInfo[1];

                double[] predictions = beMf.predict(testUser);
                double mae = new MAE(beMf).getScore(testUser, predictions);

                if (!Double.isNaN(mae)) {
                    maeByGender.put(gender, maeByGender.get(gender) + mae);
                    countByGender.put(gender, countByGender.get(gender) + 1);
                }
            } else {
                continue;
            }
        }

        // Print the average MAE for each gender
        for (String gender : maeByGender.keySet()) {
            double totalMae = maeByGender.get(gender);
            int count = countByGender.get(gender);
            double averageMae = count == 0 ? Double.NaN : totalMae / count;
            System.out.printf("Average MAE for gender %s: %.4f\n", gender, averageMae);
        }

    }

    public void runbemfseperatelytrained(DataModel datamodel,DataModel dataModelM,DataModel dataModelF) throws IOException {
        BeMF beMf;
        double[] ratings = {1.0, 2.0, 3.0, 4.0, 5.0};

        beMf = new BeMF(datamodel,2,100,0.006,0.16,ratings);
        beMf.fit();
        BeMF bemf1 = new BeMF(datamodel,2,100,0.006,0.16,ratings);
        beMf.fit();


        Map<String, Double> maeByGender = new HashMap<>();
        maeByGender.put("M", 0.0);
        maeByGender.put("F", 0.0);

        // Map to store counts for each gender
        Map<String, Integer> countByGender = new HashMap<>();
        countByGender.put("M", 0);
        countByGender.put("F", 0);
        double totalMae = 0.0;
        int count=0;
        for (TestUser testUser : dataModelM.getTestUsers()) {
            String userId = testUser.getId();
            String[] userInfo = datamodel.getDataBank().getStringArray(userId);


            if (userInfo != null) {
                String gender = userInfo[1];
                if (gender.equals("M")) {
                    double[] predictions = beMf.predict(testUser);
                    double mae = new MAE(beMf).getScore(testUser, predictions);
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
        maeByGender.put("M", totalMae/count);

        for (TestUser testUser : dataModelF.getTestUsers()) {
            String userId = testUser.getId();
            String[] userInfo = datamodel.getDataBank().getStringArray(userId);

            if (userInfo != null) {
                String gender = userInfo[1];
                if (gender.equals("F")) {
                    double[] predictions = bemf1.predict(testUser);
                    double mae = new MAE(bemf1).getScore(testUser, predictions);
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

        maeByGender.put("F", totalMae/count);

        for (String gender : maeByGender.keySet()) {

            double averageMae = maeByGender.get(gender);
            System.out.printf("Average MAE for gender %s: %.4f\n", gender, averageMae);
        }




    }


}
