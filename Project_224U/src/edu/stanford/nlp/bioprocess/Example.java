package edu.stanford.nlp.bioprocess;

import java.io.Serializable;

import edu.stanford.nlp.pipeline.Annotation;

public class Example implements Serializable {
  /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
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
