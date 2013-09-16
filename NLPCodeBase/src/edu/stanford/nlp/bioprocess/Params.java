package edu.stanford.nlp.bioprocess;

import java.io.Serializable;

import edu.stanford.nlp.util.Index;

public class Params implements Serializable {
  /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
double[][] weights;
  Index<String> featureIndex, labelIndex;
  int numUpdates;
  
  public Params() {
  }
  
  public double[][] getWeights() {
    return weights;
  }
  
  public void setWeights(double[][] wt) {
    weights = wt;
  }
  
  public Index<String> getFeatureIndex() {
	return featureIndex;
  }

  public void setFeatureIndex(Index<String> featureIndex) {
	this.featureIndex = featureIndex;
  }

  public Index<String> getLabelIndex() {
	return labelIndex;
  }

  public void setLabelIndex(Index<String> labelIndex) {
	this.labelIndex = labelIndex;
  }

  public void write(String fileName) throws Exception {
    throw new Exception("Method not implemented");
  }
  
}
