package org.example;

import au.com.bytecode.opencsv.CSVReader;
import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TrainTestFilesDataSet;

import java.io.FileReader;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {


        DataModel datamodel = BenchmarkDataModels.MovieLens1M();
        String csvFile = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\users.csv";
        datamodel=loadUserAgesFromCSV(csvFile,datamodel);

        String trainFile = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\balancedbyagegroup.csv";
        String testFile = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\abc-test.csv";
        TrainTestFilesDataSet trainTestFilesDataSet = new TrainTestFilesDataSet(trainFile, testFile);
        DataModel datamodel1 = new DataModel(trainTestFilesDataSet);

        Knn knn = new Knn();
        knn.runKnn(datamodel,datamodel1);
        Pmf pmf = new Pmf();
        pmf.runPmf(datamodel,datamodel1);
        BiasedMf biasedMf=new BiasedMf();
        biasedMf.runBiasedMf(datamodel,datamodel1);
        BeMf beMf=new BeMf();
        beMf.runBeMf(datamodel,datamodel1);
        BnMf bnMf=new BnMf();
        bnMf.runBnMf(datamodel,datamodel1);
        Nmf nmf=new Nmf();
        nmf.runNMF(datamodel,datamodel1);


//        datamodel.exportToCSV("C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\","abc");





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