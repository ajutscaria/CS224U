package edu.stanford.nlp.bioprocess;

import java.io.File;
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
	boolean useDev = false, useOneLoop = true, refreshDataFile = true;
	//useDev = true;
	useOneLoop = false;
	refreshDataFile = false;
	String examplesFileName = "trainExamples.data";
    BioprocessDataset dataset = new BioprocessDataset(groups);
    CrossValidationSplit split = null;
    int NumCrossValidation = 10;
    
    if(!useDev) {
	    File f = new File(examplesFileName);
	    if(f.exists() && !refreshDataFile)
	    	dataset.allExamples.put("train", Utils.readFile(examplesFileName));
	    else {
	    	dataset.read("train");
	    	Utils.writeFile(dataset.examples("train"), examplesFileName);
	    }
	    split = new CrossValidationSplit((ArrayList<Example>) dataset.examples("train"), NumCrossValidation);
    }
    else{
    	dataset.read("dev");
    }
    
    double sum = 0.0;
    for(int i = 1; i <= NumCrossValidation; i++) {
    	if(useDev) {
    		Learner learner = new EntityPredictionLearner();
    		Params param = learner.learn(dataset.examples("dev"));
            double f1 = 0; 
            EntityPredictionInference inferer = new EntityPredictionInference();
	    	f1 = inferer.Infer(dataset.examples("dev"), param);
            System.out.println("F1 score: " + f1);
            sum+=f1;
            break;
    	}
    	else {
	    	System.out.println("Iteration: "+i);
	    	EntityPredictionLearner learner = new EntityPredictionLearner();
	    	Params param = learner.learn(split.GetTrainExamples(i));
	    	double f1 = 0;
	    	EntityPredictionInference inferer = new EntityPredictionInference();
	    	f1 = inferer.Infer(split.GetTestExamples(i), param);
	    	sum += f1;
    	}
    	if(useOneLoop)
    		break;
    }
    if(!useDev) {
	    double average = sum/NumCrossValidation;
	    System.out.println("Average Score: "+average);
    }
    //Scorer.scoreEntityPrediction(dataset.examples("dev"));
  }
  
  public void runEventPrediction(HashMap<String, String> groups) {
	    boolean useDev = false, useOneLoop = false, refreshDataFile = false;
	    //useOneLoop = true;
		String examplesFileName = "trainExamples.data";
	    BioprocessDataset dataset = new BioprocessDataset(groups);
	    CrossValidationSplit split = null;
	    int NumCrossValidation = 10;
	    
	    if(!useDev) {
		    File f = new File(examplesFileName);
		    if(f.exists() && !refreshDataFile)
		    	dataset.allExamples.put("train", Utils.readFile(examplesFileName));
		    else {
		    	dataset.read("train");
		    	Utils.writeFile(dataset.examples("train"), examplesFileName);
		    }
		    split = new CrossValidationSplit((ArrayList<Example>) dataset.examples("train"), NumCrossValidation);
	    }
	    else{
	    	dataset.read("dev");
	    }
	    
	    double sum = 0.0;
	    for(int i = 1; i <= NumCrossValidation; i++) {
	    	if(useDev) {
	    		Learner learner = new LearnerEvent();
	    		Params params = learner.learn(dataset.examples("dev"));
	    		TriggerPredictionInference inferer = new TriggerPredictionInference();
	            double f1 = inferer.Infer(dataset.examples("dev"), params);
	            System.out.println("F1 score: " + f1);
	            sum+=f1;
	            break;
	    	}
	    	else {
		    	System.out.println("Iteration: "+i);
		    	LearnerEvent learner = new LearnerEvent();
	    		Params params = learner.learn(split.GetTrainExamples(i));
		    	double f1 = 0;
		    	TriggerPredictionInference inferer = new TriggerPredictionInference();
	            f1 = inferer.Infer(split.GetTestExamples(i), params);
		    	sum += f1;
	    	}
	    	if(useOneLoop)
	    		break;
	    }
	    if(!useDev && !useOneLoop) {
		    double average = sum/NumCrossValidation;
		    System.out.println("Average Score: "+average);
	    }
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
    if(args.length > 0 && args[0].toLowerCase().equals("-entity"))
    	new Main().runEntityPrediction(folders);
    if(args.length > 0 && args[0].equals("-event"))
    	new Main().runEventPrediction(folders);
  }
}
