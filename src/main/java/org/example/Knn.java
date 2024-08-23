package org.example;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.knn.UserKNN;
import es.upm.etsisi.cf4j.recommender.knn.userSimilarityMetric.*;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Knn {

    public void runknn(DataModel datamodel,DataModel dataModel1) throws IOException {
        UserSimilarityMetric pip = new PIP();
        UserSimilarityMetric jmsd = new JMSD();
        UserSimilarityMetric cosine = new Cosine();
        UserSimilarityMetric correlation = new Correlation();
        double[] relevantRatings = {4.0, 5.0};
        double[] nonRelevantRatings = {1.0, 2.0, 3.0};
        UserSimilarityMetric singularity = new Singularities(relevantRatings, nonRelevantRatings);
        findAverageMAE(datamodel,dataModel1, pip);
        findAverageMAE(datamodel,dataModel1, jmsd);
        findAverageMAE(datamodel,dataModel1,cosine);
        findAverageMAE(datamodel, dataModel1,correlation);
        findAverageMAE(datamodel,dataModel1, singularity);
    }

    public void runknnseperatelytrained(DataModel datamodel, DataModel dataModelM, DataModel dataModelF) throws IOException {
        UserSimilarityMetric pip = new PIP();
        UserSimilarityMetric jmsd = new JMSD();
        UserSimilarityMetric cosine = new Cosine();
        UserSimilarityMetric correlation = new Correlation();
        double[] relevantRatings = {4.0, 5.0};
        double[] nonRelevantRatings = {1.0, 2.0, 3.0};
        UserSimilarityMetric singularity = new Singularities(relevantRatings, nonRelevantRatings);
        findAverageMAE1(datamodel, pip);
        findAverageMAE1(datamodel, jmsd);
        findAverageMAE1(datamodel, cosine);
        findAverageMAE1(datamodel, correlation);
        findAverageMAE1(datamodel, singularity);
    }

    private static void findAverageMAE(DataModel datamodel,DataModel dataModel1, UserSimilarityMetric metric) {
        UserKNN userKNN = new UserKNN(dataModel1, 75, metric, UserKNN.AggregationApproach.WEIGHTED_MEAN);
        userKNN.fit();

        Map<String, Double> maeByGender = new HashMap<>();
        maeByGender.put("M", 0.0);
        maeByGender.put("F", 0.0);

        Map<String, Integer> countByGender = new HashMap<>();
        countByGender.put("M", 0);
        countByGender.put("F", 0);

        for (TestUser testUser : dataModel1.getTestUsers()) {
            String userId = testUser.getId();
            String[] userInfo = datamodel.getDataBank().getStringArray(userId);

            if (userInfo != null && userInfo.length > 1) {
                String gender = userInfo[1];
                double[] predictions = userKNN.predict(testUser);
                double mae = new MAE(userKNN).getScore(testUser, predictions);

                if (!Double.isNaN(mae)) {
                    maeByGender.put(gender, maeByGender.get(gender) + mae);
                    countByGender.put(gender, countByGender.get(gender) + 1);
                }
            }
        }

        for (String gender : maeByGender.keySet()) {
            double totalMae = maeByGender.get(gender);
            int count = countByGender.get(gender);
            double averageMae = count == 0 ? Double.NaN : totalMae / count;
            System.out.printf("Average MAE for gender %s: %.4f\n", gender, averageMae);
        }
    }
    private static void findAverageMAE1(DataModel datamodel, UserSimilarityMetric metric) {
        UserKNN userKNN = new UserKNN(datamodel, 75, metric, UserKNN.AggregationApproach.MEAN);
        userKNN.fit();

        Map<String, Double> maeByGender = new HashMap<>();
        maeByGender.put("M", 0.0);
        maeByGender.put("F", 0.0);

        Map<String, Integer> countByGender = new HashMap<>();
        countByGender.put("M", 0);
        countByGender.put("F", 0);

        for (TestUser testUser : datamodel.getTestUsers()) {
            String userId = testUser.getId();
            String[] userInfo = datamodel.getDataBank().getStringArray(userId);

            if (userInfo != null && userInfo.length > 1) {
                String gender = userInfo[1];
                double[] predictions = userKNN.predict(testUser);
                double mae = new MAE(userKNN).getScore(testUser, predictions);

                if (!Double.isNaN(mae)) {
                    maeByGender.put(gender, maeByGender.get(gender) + mae);
                    countByGender.put(gender, countByGender.get(gender) + 1);
                }
            }
        }

        for (String gender : maeByGender.keySet()) {
            double totalMae = maeByGender.get(gender);
            int count = countByGender.get(gender);
            double averageMae = count == 0 ? Double.NaN : totalMae / count;
            System.out.printf("Average MAE for gender %s: %.4f\n", gender, averageMae);
        }
    }
}
