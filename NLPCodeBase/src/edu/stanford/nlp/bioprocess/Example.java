package edu.stanford.nlp.bioprocess;

import edu.stanford.nlp.pipeline.Annotation;

public class Example {
  String id, data;
  Annotation gold, prediction;
  
  public String getData() {
    return data;
  }
  
  public Annotation getGold() {
    return gold;
  }
  
  public void setPrediction(Annotation pred) {
    prediction = pred;
  }
}
