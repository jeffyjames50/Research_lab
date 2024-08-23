package org.example;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.*;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class BnMf {

    public void runBnmf(DataModel datamodel,DataModel dataModel1) throws IOException {
        BNMF bnMf;


        bnMf = new BNMF(dataModel1,10,100,0.6,5);
        bnMf.fit();

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

                double[] predictions = bnMf.predict(testUser);
                double mae = new MAE(bnMf).getScore(testUser, predictions);

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

    public void runbnmfseperatelytrained(DataModel datamodel,DataModel dataModelM,DataModel dataModelF) throws IOException {
        BNMF bnmf;


        bnmf = new BNMF(dataModelM,10,100,0.6,5);
        bnmf.fit();
        BNMF bnmf1 = new BNMF(dataModelF,10,100,0.6,5);
        bnmf1.fit();


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
                    double[] predictions = bnmf.predict(testUser);
                    double mae = new MAE(bnmf).getScore(testUser, predictions);
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
                    double[] predictions = bnmf1.predict(testUser);
                    double mae = new MAE(bnmf1).getScore(testUser, predictions);
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
