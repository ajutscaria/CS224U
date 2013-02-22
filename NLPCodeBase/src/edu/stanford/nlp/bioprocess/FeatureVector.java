package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;

import edu.stanford.nlp.util.Pair;

public class FeatureVector {
  ArrayList<String> indicatorFeatures;
  ArrayList<Pair<String, Double>> generalFeatures;
  ArrayList<Pair<String, Integer>> maxExamples;
  
  public FeatureVector() {
    indicatorFeatures = new ArrayList<String>();
    generalFeatures = new ArrayList<Pair<String, Double>>();
    maxExamples = new ArrayList<Pair<String, Integer>>();
  }
  
  public void add(String indicator) {
    indicatorFeatures.add(indicator);
  }
  
  public void add(String feature, double value) {
    generalFeatures.add(new Pair<String, Double>(feature, value));
  }
  
  public void add(FeatureVector fv) throws Exception {
    throw new Exception("Method not implemented");
  }
  
  public double dotProduct(Params param) throws Exception {
    throw new Exception("Method not implemented");
  }
  
}
