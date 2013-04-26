package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

public class FeatureVector {
  List<String> features;
  
  public FeatureVector() {
	  features = new ArrayList<String>();
  }
  
  public FeatureVector(List<String> features) {
	  this.features = features;
  }
  
  public void add(String feature) {
	  features.add(feature);
  }  
  
  public List<String> getFeatures(){
	  return features;
  }
  
  public void add(List<String> feature) {
	  features.addAll(feature);
  }

  public String getFeatureString() {
	  StringBuilder f = new StringBuilder();
	  for(String feature:features)
		  f.append("\n\t" + feature);
	return f.toString();
  }
}

