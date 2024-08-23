package org.example;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.*;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class Nmf {

    public void runnmf(DataModel datamodel,DataModel dataModel1) throws IOException {
        NMF nmf;


        nmf = new NMF(dataModel1,2,100);
        nmf.fit();


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

                double[] predictions = nmf.predict(testUser);
                double mae = new MAE(nmf).getScore(testUser, predictions);

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

    public void runnmfseperatelytrained(DataModel datamodel,DataModel dataModelM,DataModel dataModelF) throws IOException {
        NMF nmf;


        nmf = new NMF(dataModelM,2,100);
        nmf.fit();
        NMF nmf1 = new NMF(dataModelF,2,100);
        nmf1.fit();


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
                    double[] predictions = nmf.predict(testUser);
                    double mae = new MAE(nmf).getScore(testUser, predictions);
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
                    double[] predictions = nmf1.predict(testUser);
                    double mae = new MAE(nmf1).getScore(testUser, predictions);
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
