package edu.stanford.nlp.bioprocess;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Properties;

import edu.stanford.nlp.util.StringUtils;

public class Main {
  /***
   * 
   * @param groups - The types of data groups that will be used, e.g. test and train.
   */
  public void runEntityPrediction(HashMap<String, String> groups) {
	String examplesFileName = "trainExamples.data";
    BioprocessDataset dataset = new BioprocessDataset(groups);
    File f = new File(examplesFileName);
    if(f.exists())
    	dataset.allExamples.put("train", Utils.readFile(examplesFileName));
    else {
    	dataset.read("train");
    	Utils.writeFile(dataset.examples("train"), examplesFileName);
    }
    System.out.println(dataset.examples("train").size());
    int NumCrossValidation = 10;
    CrossValidationSplit split = new CrossValidationSplit((ArrayList<Example>) dataset.examples("train"), NumCrossValidation);
    double sum = 0.0;
    for(int i = 1; i <= NumCrossValidation; i++) {
    	System.out.println("Iteration: "+i);
    	Learner learner = new Learner(split.GetTrainExamples(i));
    	double f1 = learner.learnAndPredict(split.GetTestExamples(i));
    	//Learner learner = new Learner(dataset.examples("dev"));
        //double f1 = learner.learnAndPredict(dataset.examples("dev"));
        //EntityPredictionInference infer = new EntityPredictionInference();

        //if (!Double.isNaN(f1)) {
        	sum += f1;
        //}
    }
    double average = sum/NumCrossValidation;
    System.out.println("Average Score: "+average);
    //Scorer.scoreEntityPrediction(dataset.examples("dev"));
  }
  
  public void runEventPrediction(HashMap<String, String> groups) {
    BioprocessDataset dataset = new BioprocessDataset(groups);
    dataset.readAll();
    //Learner learner = new Learner(dataset.examples("train"));
    //learner.learn();
    TriggerPredictionInference TrigPred = new TriggerPredictionInference();
    TrigPred.baselineInfer(dataset.examples("train"));
    //Scorer.scoreEventPrediction(dataset.examples("train"));
  }
  
  /***
   * Entry point to the bio process project. 
   * @param args
   */
  public static void main(String[] args) {
    Properties props = StringUtils.propFileToProperties("src/edu/stanford/nlp/bioprocess/bioprocess.properties");
    String trainDirectory = props.getProperty("train.dir"), testDirectory = props.getProperty("test.dir"),
    		devDirectory = props.getProperty("dev.dir");
    HashMap<String, String> folders = new HashMap<String, String>();
    folders.put("test", testDirectory);
    folders.put("train", trainDirectory);
    folders.put("dev", devDirectory);
    if(args.length > 0 && args[0].equals("-entity"))
    	new Main().runEntityPrediction(folders);
    if(args.length > 0 && args[0].equals("-event"))
    	new Main().runEventPrediction(folders);
  }
}
