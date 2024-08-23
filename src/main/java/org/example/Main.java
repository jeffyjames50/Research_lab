package org.example;


import au.com.bytecode.opencsv.CSVReader;
import es.upm.etsisi.cf4j.data.*;

import java.io.FileReader;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {


        DataModel datamodel = BenchmarkDataModels.MovieLens1M();
        String csvFile = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\users.csv";
        datamodel=loadUserAgesFromCSV(csvFile,datamodel);

        String trainFile = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\balancedbygendergroup.csv";
        String testFile = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\abc-test.csv";
        TrainTestFilesDataSet trainTestFilesDataSet = new TrainTestFilesDataSet(trainFile, testFile);
        DataModel datamodel1 = new DataModel(trainTestFilesDataSet);

        System.out.println(" MAE of the predictions by gender groups");
        Pmf pmf = new Pmf();
        pmf.runPmf(datamodel,datamodel);

        Biasedmf biasedmf = new Biasedmf();
        biasedmf.runBiasedmf(datamodel,datamodel);

        BeMf beMF = new BeMf();
        beMF.runBemf(datamodel,datamodel);

        BnMf bnMf = new BnMf();
        bnMf.runBnmf(datamodel,datamodel);

        Nmf nmf = new Nmf();
        nmf.runnmf(datamodel,datamodel);

        Knn knn= new Knn();
        knn.runknn(datamodel,datamodel);



        System.out.println("MAE of the predictions by balanced gender groups");
        pmf.runPmf(datamodel,datamodel1);
        biasedmf.runBiasedmf(datamodel,datamodel1);
        beMF.runBemf(datamodel,datamodel1);
        bnMf.runBnmf(datamodel,datamodel1);
        nmf.runnmf(datamodel,datamodel1);
        knn.runknn(datamodel,datamodel1);

        System.out.println("MAE of predictions from separately trained gender groups. Imbalanced groups.");
        SeperatelyTrainedGenderGroup seperatelyTrainedGenderGroup=new SeperatelyTrainedGenderGroup();
        seperatelyTrainedGenderGroup.runSeperatelyTrainedGender(datamodel);

        System.out.println("MAE of predictions from separately trained gender groups. Balanced groups");
        Septrainedbalanced seperatelyTrainedBalanced=new Septrainedbalanced();
        seperatelyTrainedBalanced.runSeperatelyTrainedbalanced(datamodel);

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