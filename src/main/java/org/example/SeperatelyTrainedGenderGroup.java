package org.example;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TrainTestFilesDataSet;

import java.io.IOException;

public class SeperatelyTrainedGenderGroup {

    public void runSeperatelyTrainedGender(DataModel datamodel) throws IOException {

        String trainFileM = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\users_groupM.csv";
        String trainFileF = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\users_groupF.csv";


        String testFile = "C:\\Users\\JERRY\\Documents\\web and ds\\Research Lab\\ml-1m\\abc-test.csv";

        TrainTestFilesDataSet trainTestFilesDataSetM = new TrainTestFilesDataSet(trainFileM, testFile);
        DataModel datamodelM = new DataModel(trainTestFilesDataSetM);

        TrainTestFilesDataSet trainTestFilesDataSetF = new TrainTestFilesDataSet(trainFileF, testFile);
        DataModel datamodelF = new DataModel(trainTestFilesDataSetF);

//        Pmf pmf = new Pmf();
//        pmf.runPmf(datamodel,datamodelM);
//        pmf.runPmf(datamodel,datamodelF);
//
//
        Biasedmf biasedmf = new Biasedmf();
        biasedmf.runbiasedmfseperatelytrained(datamodel,datamodelM,datamodelF);


        BeMf beMF = new BeMf();
        beMF.runbemfseperatelytrained(datamodel,datamodelM,datamodelF);

        BnMf bnMf = new BnMf();
        bnMf.runbnmfseperatelytrained(datamodel,datamodelM,datamodelF);

        Nmf nmf = new Nmf();
        nmf.runnmfseperatelytrained(datamodel,datamodelM,datamodelF);

        Pmf pmf=new Pmf();
        pmf.runpmfseperatelytrained(datamodel,datamodelM,datamodelF);

        Knn knn= new Knn();
        knn.runknnseperatelytrained(datamodel,datamodelM,datamodelF);





    }
}
