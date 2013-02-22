package edu.stanford.nlp.bioprocess;

import java.util.HashMap;
import java.util.Properties;

import edu.stanford.nlp.util.StringUtils;

public class Main {
  public void run(HashMap<String, String> groups) {
    BioprocessDataset dataset = new BioprocessDataset();
    for(String group : groups.keySet())
      dataset.addGroup(group, groups.get(group));
    dataset.read();
    Learner learner = new Learner(dataset.examples("train"));
    learner.learn();
  }
  
  public static void main(String[] args) {
    Properties props = StringUtils.propFileToProperties("src/edu/stanford/nlp/bioprocess/bioprocess.properties");
    String trainDirectory = props.getProperty("train.dir"), testDirectory = props.getProperty("test.dir");
    HashMap<String, String> folders = new HashMap<String, String>();
    folders.put("test", testDirectory);
    folders.put("train", trainDirectory);
    new Main().run(folders);
  }
}
