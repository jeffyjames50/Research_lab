package org.example;

import au.com.bytecode.opencsv.CSVParser;
import au.com.bytecode.opencsv.CSVReader;
import es.upm.etsisi.cf4j.data.*;
import es.upm.etsisi.cf4j.data.types.DataSetEntry;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.cf4j.recommender.knn.UserKNN;
import es.upm.etsisi.cf4j.recommender.knn.userSimilarityMetric.*;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;

import java.io.FileReader;
import java.util.*;
import java.io.IOException;

import java.util.HashMap;
import java.util.Map;

public class Main {

    public static void main(String[] args) {
        try {
            // Load the MovieLens 1M dataset

            DataModel datamodel = BenchmarkDataModels.MovieLens1M();
            String csvFile = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\users.csv";
            datamodel=loadUserAgesFromCSV(csvFile,datamodel);


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





            String trainFile1 = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\users_group1.csv";
            String trainFile18 = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\users_group18.csv";
            String trainFile25 = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\users_group25.csv";
            String trainFile35 = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\users_group35.csv";
            String trainFile45 = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\users_group45.csv";
            String trainFile50 = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\users_group50.csv";
            String trainFile56 = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\users_group56.csv";




            String testFile = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\abc-test.csv";
                        //System.out.println(totalMae);
            //System.out.println(count);

            Pmf pmf= new Pmf();
            Map<String, Double> maeByAgeGroup1=pmf.pmfAgeGroup(trainFile1,trainFile18,trainFile25,trainFile35,trainFile45,trainFile50,trainFile56,maeByAgeGroup,testFile,datamodel);
            printOutput(maeByAgeGroup1);

            UserSimilarityMetric pip=new PIP();
            UserSimilarityMetric jmsd=new JMSD();
            UserSimilarityMetric cosine=new Cosine();
            UserSimilarityMetric correlation=new Correlation();
            double[] relevantRatings = { 4.0, 5.0};
            double[] nonRelevantRatings = {1.0,2.0,3.0};
            UserSimilarityMetric singularity=new Singularities(relevantRatings,nonRelevantRatings);

            Knn knn= new Knn();
            Map<String, Double> maeByAgeGroup2=knn.knnAgeGroup(trainFile1,trainFile18,trainFile25,trainFile35,trainFile45,trainFile50,trainFile56,maeByAgeGroup,testFile,datamodel,pip);
            printOutput(maeByAgeGroup2);


            Map<String, Double> maeByAgeGroup3=knn.knnAgeGroup(trainFile1,trainFile18,trainFile25,trainFile35,trainFile45,trainFile50,trainFile56,maeByAgeGroup,testFile,datamodel,jmsd);
            printOutput(maeByAgeGroup3);


            Map<String, Double> maeByAgeGroup4=knn.knnAgeGroup(trainFile1,trainFile18,trainFile25,trainFile35,trainFile45,trainFile50,trainFile56,maeByAgeGroup,testFile,datamodel,cosine);
            printOutput(maeByAgeGroup4);

            Map<String, Double> maeByAgeGroup5=knn.knnAgeGroup(trainFile1,trainFile18,trainFile25,trainFile35,trainFile45,trainFile50,trainFile56,maeByAgeGroup,testFile,datamodel,correlation);
            printOutput(maeByAgeGroup5);

            Map<String, Double> maeByAgeGroup6=knn.knnAgeGroup(trainFile1,trainFile18,trainFile25,trainFile35,trainFile45,trainFile50,trainFile56,maeByAgeGroup,testFile,datamodel,singularity);
            printOutput(maeByAgeGroup6);

            Biasedmf biasedmf= new Biasedmf();
            Map<String, Double> maeByAgeGroup7=biasedmf.biasedmfAgeGroup(trainFile1,trainFile18,trainFile25,trainFile35,trainFile45,trainFile50,trainFile56,maeByAgeGroup,testFile,datamodel);
            printOutput(maeByAgeGroup7);

            Bemf bemf= new Bemf();
            Map<String, Double> maeByAgeGroup8=bemf.bemfAgeGroup(trainFile1,trainFile18,trainFile25,trainFile35,trainFile45,trainFile50,trainFile56,maeByAgeGroup,testFile,datamodel);
            printOutput(maeByAgeGroup8);

            BnMf bnmf= new BnMf();
            Map<String, Double> maeByAgeGroup9=bnmf.bnmfAgeGroup(trainFile1,trainFile18,trainFile25,trainFile35,trainFile45,trainFile50,trainFile56,maeByAgeGroup,testFile,datamodel);
            printOutput(maeByAgeGroup9);

            Nmf nmf= new Nmf();
            Map<String, Double> maeByAgeGroup10=nmf.nmfAgeGroup(trainFile1,trainFile18,trainFile25,trainFile35,trainFile45,trainFile50,trainFile56,maeByAgeGroup,testFile,datamodel);
            printOutput(maeByAgeGroup10);



    } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    private static void printOutput(Map<String, Double> maeByAgeGroup){
        for (String ageGroup : maeByAgeGroup.keySet()) {
            System.out.printf("Average MAE for age group %s: %.4f\n", ageGroup, maeByAgeGroup.get(ageGroup));
        }

    }


    private static DataModel loadUserAgesFromCSV(String csvFile, DataModel dataModel) throws IOException {
        try (CSVReader reader = new CSVReader(new FileReader(csvFile))) {
            String[] nextLine;
            while ((nextLine = reader.readNext()) != null) {
                // Assuming the CSV format is userId, gender, age, occupation, zipCode
                // Change the index accordingly if the format is different
                String userId = nextLine[0];
                String ageStr = nextLine[2]; // Age is at index 2
                String gender = nextLine[1];
                String[] arr= {ageStr,gender};
                dataModel.getDataBank().setStringArray(userId,arr);
            }
        }

        return dataModel;

    }


}



