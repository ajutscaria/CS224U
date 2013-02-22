package edu.stanford.nlp.bioprocess;

import java.util.HashMap;

public class Params {
  HashMap<String, Double> weights;
  int numUpdates;
  
  public Params() {
    weights = new HashMap<String, Double>();
  }
  
  public void read(String fileName) throws Exception {
    throw new Exception("Method not implemented");
  }
  
  public void update(HashMap<String, Double> wts)  throws Exception {
    throw new Exception("Method not implemented");
  }
  
  public double getWeight(String feature) {
    return weights.get(feature);
  }
  
  public void setWeight(String feature, double weight) {
    weights.put(feature, weight);
  }
  
  public void write(String fileName) throws Exception {
    throw new Exception("Method not implemented");
  }
  
}
