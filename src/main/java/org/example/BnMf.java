package org.example;
import au.com.bytecode.opencsv.CSVReader;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.knn.UserKNN;
import es.upm.etsisi.cf4j.recommender.knn.userSimilarityMetric.PIP;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BNMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BeMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;

import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
public class BnMf {

    public void runBnMf(DataModel datamodel) throws IOException {
        BNMF bnMf;


        bnMf = new BNMF(datamodel,10,100,0.6,5);
        bnMf.fit();

        Map<String, int[]> ageGroups = new HashMap<>();
        ageGroups.put("1", new int[]{1, 17});
        ageGroups.put("18", new int[]{18, 24});
        ageGroups.put("25", new int[]{25, 34});
        ageGroups.put("35", new int[]{35, 44});
        ageGroups.put("45", new int[]{45, 49});
        ageGroups.put("50", new int[]{50, 55});
        ageGroups.put("56", new int[]{56, 100});

        // Map to store MAE values for each age group
        Map<String, Double> maeByAgeGroup = new TreeMap<>();
        for (String ageGroup : ageGroups.keySet()) {
            maeByAgeGroup.put(ageGroup, 0.0);
        }


        // Map to store counts for each age group
        Map<String, Integer> countByAgeGroup = new TreeMap<>();
        for (String ageGroup : ageGroups.keySet()) {
            countByAgeGroup.put(ageGroup, 0);
        }

        for (TestUser testUser : datamodel.getTestUsers()) {
            String userId1 = testUser.getId(); // User IDs start from 1
            Integer ageInteger = Integer.parseInt(datamodel.getDataBank().getStringArray(userId1)[0]);
//            if (ageInteger == null) {
//                continue; // Skip processing for this user
//            }
            int age = ageInteger.intValue();
            for (String ageGroup : ageGroups.keySet()) {
                int[] range = ageGroups.get(ageGroup);
                if (age >= range[0] && age <= range[1]) {
                    double[] predictions = bnMf.predict(testUser);
                    double mae = new MAE(bnMf).getScore(testUser, predictions);

                    if (!Double.isNaN(mae)) {
                        maeByAgeGroup.put(ageGroup, maeByAgeGroup.get(ageGroup) + mae);
                        countByAgeGroup.put(ageGroup, countByAgeGroup.get(ageGroup) + 1);
                    }
                    break;
                }
            }
        }

        //System.out.println(maeByAgeGroup);
        //System.out.println(countByAgeGroup);

        // Iterate over each test user and calculate the MAE for each age group

        // Print the average MAE for each age group
        for (String ageGroup : maeByAgeGroup.keySet()) {
            double totalMae = maeByAgeGroup.get(ageGroup);
            int count = countByAgeGroup.get(ageGroup);
            //System.out.println("count : "+count);
            //System.out.println("totalmae : "+totalMae);
            double averageMae = count == 0 ? Double.NaN : totalMae / count;
            System.out.printf("Average MAE for age group %s: %.4f\n", ageGroup, averageMae);
        }

    }

}
