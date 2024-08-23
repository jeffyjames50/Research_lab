package org.example;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.knn.UserKNN;
import es.upm.etsisi.cf4j.recommender.knn.userSimilarityMetric.*;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class Knn {

    public void runKnn(DataModel datamodel,DataModel dm1 ) throws IOException {

        //userKNN = new UserKNN(datamodel, 75,new Correlation(), UserKNN.AggregationApproach.MEAN);
        //userKNN = new UserKNN(datamodel, 75,new Cosine(), UserKNN.AggregationApproach.MEAN);
        //userKNN = new UserKNN(datamodel, 75,new JMSD(), UserKNN.AggregationApproach.MEAN);
        UserSimilarityMetric pip=new PIP();
        UserSimilarityMetric jmsd=new JMSD();
        UserSimilarityMetric cosine=new Cosine();
        UserSimilarityMetric correlation=new Correlation();
        double[] relevantRatings = { 4.0, 5.0};
        double[] nonRelevantRatings = {1.0,2.0,3.0};
        UserSimilarityMetric singularity=new Singularities(relevantRatings,nonRelevantRatings);
        findAverageMAE(datamodel,pip,dm1);
        findAverageMAE(datamodel,jmsd,dm1);
        findAverageMAE(datamodel,cosine,dm1);
        findAverageMAE(datamodel,correlation,dm1);
        findAverageMAE(datamodel,singularity,dm1);



    }

    private static void findAverageMAE(DataModel datamodel,UserSimilarityMetric metric,DataModel dm1){
        UserKNN userKNN;
        userKNN = new UserKNN(dm1, 75,metric, UserKNN.AggregationApproach.WEIGHTED_MEAN);
        userKNN.fit();

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
            int age = ageInteger;
            for (String ageGroup : ageGroups.keySet()) {
                int[] range = ageGroups.get(ageGroup);
                if (age >= range[0] && age <= range[1]) {
                    double[] predictions = userKNN.predict(testUser);
                    double mae = new MAE(userKNN).getScore(testUser, predictions);

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
