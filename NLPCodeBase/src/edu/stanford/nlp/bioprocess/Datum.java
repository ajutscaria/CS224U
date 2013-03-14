package edu.stanford.nlp.bioprocess;

import java.util.List;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

public class Datum {
  CoreMap sentence;
  public final String word;
  public final String label;
  FeatureVector features;
  public List<String> getFeatures() {
	return features.getFeatures();
  }

  public void setFeatures(FeatureVector features) {
	this.features = features;
  }
 
  public String guessLabel;
  Tree entityNode, eventNode;
  IndexedWord entityHead, eventHead;
  double probEntity;
  double[] probSRL = new double[ArgumentRelation.getSemanticRoles().size()];
  
  public Datum(CoreMap sentence, String word, String label, Tree entityNode, Tree eventNode) {
	this.sentence = sentence;
    this.word = word;
    this.label = label;
    this.entityNode = entityNode;
    this.eventNode = eventNode;
  }
  
  public void setProbability(double prob) {
	  probEntity = prob;
  }
  
  public double getProbability() {
	  return probEntity;
  }
 
  public void setProbabilitySRL(double[] probSRL) {
	  this.probSRL = probSRL;
	  System.out.println(this.entityNode+":"+this.eventNode);
	  for (int i = 0; i<probSRL.length; i++) {
		  System.out.print(probSRL[i]+" ");
	  }
	  System.out.println();
  }
}