package edu.stanford.nlp.bioprocess;

import java.io.Serializable;

import edu.stanford.nlp.pipeline.Annotation;

public class Example implements Serializable {
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
