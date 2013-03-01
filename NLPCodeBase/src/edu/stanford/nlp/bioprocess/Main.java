package edu.stanford.nlp.bioprocess;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

public class Main {
  /***
   * 
   * @param groups - The types of data groups that will be used, e.g. test and train.
   */
  public void runEntityPrediction(HashMap<String, String> groups) {
	boolean useDev = false, useOneLoop = true;
	//useDev = true;
	useOneLoop = false;
	String examplesFileName = "trainExamples.data";
    BioprocessDataset dataset = new BioprocessDataset(groups);
    CrossValidationSplit split = null;
    int NumCrossValidation = 10;
    
    if(!useDev) {
	    File f = new File(examplesFileName);
	    if(f.exists())
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
    		Learner learner = new Learner(dataset.examples("dev"));
            double f1 = learner.learnAndPredict(dataset.examples("dev"));
            break;
    	}
    	else {
	    	System.out.println("Iteration: "+i);
	    	Learner learner = new Learner(split.GetTrainExamples(i));
	    	double f1 = learner.learnAndPredict(split.GetTestExamples(i));
	    	sum += f1;
    	}
    	if(useOneLoop)
    		break;
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
    //new Main().checkDP(folders);
    
  }

	private void checkDP(HashMap<String, String> folders) {
		 BioprocessDataset dataset = new BioprocessDataset(folders);
		 dataset.readAll();
		 for (Example ex : dataset.examples("train")) {
			 for (CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
				 HashMap<Tree, Pair<Double, String>> tokenMap = new HashMap<Tree, Pair<Double, String>>();
				 Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
				 for (Tree node : syntacticParse.preOrderNodeList()) {
					 Double prob = Math.random();
					 String label;
					 if (node.isLeaf()) {
						 if (prob < 0.5) {
							 label = "O";
						 } else {
							 label = "E";
						 }
					 } else {
						 label = "NA";
					 }
					 Pair<Double, String> pair = new Pair<Double, String>(prob, label);
					 tokenMap.put(node, pair);
				 }
				 DynamicProgramming dp = new DynamicProgramming(sentence, tokenMap);
				 break;
			 }
			 break;
		 }
	}
}
