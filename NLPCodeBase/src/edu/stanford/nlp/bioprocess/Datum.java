package edu.stanford.nlp.bioprocess;

import java.util.List;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

public class Datum {
  CoreMap sentence;
  public final String word;
  public final String label;
  public final String role;
  
  FeatureVector features;
  public List<String> getFeatures() {
	return features.getFeatures();
  }

  public void setFeatures(FeatureVector features) {
	this.features = features;
  }
 
  public String guessLabel;
  public String guessRole;
  public int bestRoleIndex;
  public double bestRoleProbability;
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
    this.role = null;
  }
  
  public Datum(CoreMap sentence, String word, String label, Tree entityNode, Tree eventNode, String role) {
		this.sentence = sentence;
	    this.word = word;
	    this.label = label;
	    this.entityNode = entityNode;
	    this.eventNode = eventNode;
	    this.role = role;
  }
  
  public void setProbability(double prob) {
	  probEntity = prob;
  }
  
  public double getProbability() {
	  return probEntity;
  }
 
  public void setProbabilitySRL(double[] probSRL) {
	  this.probSRL = probSRL;
	  double maximum = Double.NEGATIVE_INFINITY;
	  bestRoleIndex = -1;
	  for (int i=0; i<this.probSRL.length; i++) {
		  System.out.print(this.probSRL[i]+" ");
		  if (this.probSRL[i] >= maximum) {
			  maximum = this.probSRL[i];
			  bestRoleIndex = i;
		  }
	  }
	  bestRoleProbability = maximum;
	  System.out.println(bestRoleIndex);
  }

public double getBestRoleProbability() {
	return bestRoleProbability;
}
  
public int getBestRoleIndex() {
	return bestRoleIndex;
}

public String getBestRole() {
	// TODO Auto-generated method stub
	return null;
}
}