package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

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
  List<Pair<String, Double>> rankedRoleProbs = new ArrayList<Pair<String, Double>>();
  
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
 
  public void setProbabilitySRL(double[] probSRL, Index labelIndex) {
	  this.probSRL = probSRL;
	  this.rankedRoleProbs = Utils.rankRoleProbs(probSRL, labelIndex);
  }
  
  public List<Pair<String, Double>> getRankedRoles() {
	  return this.rankedRoleProbs;
  }

  public double[] getProbabilitySRL() {
	  return this.probSRL;
  }
  
public double getBestRoleProbability() {
	return bestRoleProbability;
}

public String getBestRole() {
	return this.rankedRoleProbs.get(0).first;
}

}