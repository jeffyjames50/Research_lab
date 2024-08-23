package org.example;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class Pmf {

    public void runPmf(DataModel datamodel,DataModel dm1) throws IOException {
        PMF pmf;
        pmf = new PMF(dm1,8,100,0.045,0.01,3);
        pmf.fit();

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

        for (TestUser testUser : dm1.getTestUsers()) {
            String userId1 = testUser.getId(); // User IDs start from 1
            String[] userData = datamodel.getDataBank().getStringArray(userId1);

            if (userData == null || userData.length == 0) {
                continue; // Skip processing if no data available for the user
            }
            Integer ageInteger = Integer.parseInt(userData[0]);
            int age = ageInteger.intValue();
            for (String ageGroup : ageGroups.keySet()) {
                int[] range = ageGroups.get(ageGroup);
                if (age >= range[0] && age <= range[1]) {
                    double[] predictions = pmf.predict(testUser);
                    double mae = new MAE(pmf).getScore(testUser, predictions);

                    if (!Double.isNaN(mae)) {
                        maeByAgeGroup.put(ageGroup, maeByAgeGroup.get(ageGroup) + mae);
                        countByAgeGroup.put(ageGroup, countByAgeGroup.get(ageGroup)+1);
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
