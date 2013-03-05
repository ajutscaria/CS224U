package edu.stanford.nlp.bioprocess;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.util.StringUtils;

public class Main {
	
  public void runPrediction(HashMap<String, String> groups, FeatureExtractor featureFactory, Learner learner, Inferer inferer, Scorer scorer) {
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
	    		Params param = learner.learn(dataset.examples("dev"));
	            List<Datum> predicted = inferer.Infer(dataset.examples("dev"), param);
	            double f1 = Scorer.score(predicted);
	            System.out.println("F1 score: " + f1);
	            sum+=f1;
	            break;
	    	}
	    	else {
		    	System.out.println("Iteration: "+i);
		    	Params param = learner.learn(split.GetTrainExamples(i));
		    	List<Datum> predicted = inferer.Infer(split.GetTestExamples(i), param);
		    	double f1 = Scorer.score(predicted);
		    	sum += f1;
	    	}
	    	if(useOneLoop)
	    		break;
	    }
	    if(!useDev) {
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
    	new Main().runPrediction(folders, new EntityFeatureFactory(), new EntityPredictionLearner(), new EntityPredictionInferer(), new Scorer());
    if(args.length > 0 && args[0].equals("-event"))
    	new Main().runPrediction(folders, new EventFeatureFactory(), new EventPredictionLearner(), new EventPredictionInferer(), new Scorer());
  }
}
