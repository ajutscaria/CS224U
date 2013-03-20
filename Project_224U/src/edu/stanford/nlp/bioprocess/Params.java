package edu.stanford.nlp.bioprocess;

public class Params {
  double[][] weights;
  Index featureIndex, labelIndex;
  int numUpdates;
  
  public Params() {
  }
  
  public double[][] getWeights() {
    return weights;
  }
  
  public void setWeights(double[][] wt) {
    weights = wt;
  }
  
  public Index getFeatureIndex() {
	return featureIndex;
  }

  public void setFeatureIndex(Index featureIndex) {
	this.featureIndex = featureIndex;
  }

  public Index getLabelIndex() {
	return labelIndex;
  }

  public void setLabelIndex(Index labelIndex) {
	this.labelIndex = labelIndex;
  }

  public void write(String fileName) throws Exception {
    throw new Exception("Method not implemented");
  }
  
}
