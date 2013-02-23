package edu.stanford.nlp.bioprocess;

import java.util.HashMap;
import java.util.Properties;

import edu.stanford.nlp.util.StringUtils;

public class Main {
  /***
   * 
   * @param groups - The types of data groups that will be used, e.g. test and train.
   */
  public void runEntityPrediction(HashMap<String, String> groups) {
    BioprocessDataset dataset = new BioprocessDataset(groups);
    dataset.read();
    Learner learner = new Learner(dataset.examples("train"));
    learner.learn();
  }
  
  public void runEventPrediction(HashMap<String, String> groups) {
    BioprocessDataset dataset = new BioprocessDataset(groups);
    dataset.read();
    Learner learner = new Learner(dataset.examples("train"));
    learner.learn();
  }
  
  /***
   * Entry point to the bio process project. 
   * @param args
   */
  public static void main(String[] args) {
    Properties props = StringUtils.propFileToProperties("src/edu/stanford/nlp/bioprocess/bioprocess.properties");
    String trainDirectory = props.getProperty("train.dir"), testDirectory = props.getProperty("test.dir");
    HashMap<String, String> folders = new HashMap<String, String>();
    folders.put("test", testDirectory);
    folders.put("train", trainDirectory);
    if(args.length > 0 && args[0].equals("-entity"))
    	new Main().runEntityPrediction(folders);
    if(args.length > 0 && args[0].equals("-event"))
    	new Main().runEventPrediction(folders);
  }
}
