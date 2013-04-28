package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;

public class BioDatum {
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
 
  private String exampleID;
  public String guessLabel;
  public String guessRole;
  public int bestRoleIndex;
  public double bestRoleProbability;
  Tree entityNode, eventNode;
  EventMention event1, event2;
  IndexedWord entityHead, eventHead;
  double probEntity;
  //double[] probSRL = new double[ArgumentRelation.getSemanticRoles().size()];
  Counter<String> probSRL;
  List<Pair<String, Double>> rankedRoleProbs = new ArrayList<Pair<String, Double>>();
  HashMap<String, Double> rankMap = new HashMap<String, Double>();
  
  public BioDatum(CoreMap sentence, String word, String label, Tree entityNode, Tree eventNode) {
	this.sentence = sentence;
    this.word = word;
    this.label = label;
    this.entityNode = entityNode;
    this.eventNode = eventNode;
    this.role = null;
  }
  
  public BioDatum(CoreMap sentence, String word, String label, Tree entityNode, Tree eventNode, String role) {
		this.sentence = sentence;
	    this.word = word;
	    this.label = label;
	    this.entityNode = entityNode;
	    this.eventNode = eventNode;
	    this.role = role;
  }
  
  public BioDatum(CoreMap sentence, String word, String label,
		EventMention event1, EventMention event2) {
		this.sentence = sentence;
	    this.word = word;
	    this.label = label;
	    this.event1 = event1;
	    this.event2 = event2;
	    this.role = null;
}

public void setPredictedLabel(String predictedLabel) {
	this.guessLabel = predictedLabel;	
  }
  
  public String label() {
	return label;
  }
  
  public String predictedLabel() {
	return this.guessLabel;
  }

  public void setProbability(double prob) {
	  probEntity = prob;
  }
  
  public double getProbability() {
	  return probEntity;
  }
 
  public void setProbabilitySRL(Counter<String> probSRL, Index<String> labelIndex) {
	  this.probSRL = probSRL;
	  this.rankedRoleProbs = Utils.rankRoleProbs(probSRL, labelIndex);
	  for (int i=0; i<this.rankedRoleProbs.size(); i++) {
		  this.rankMap.put(this.rankedRoleProbs.get(i).first, this.rankedRoleProbs.get(i).second);
	  }
	  this.guessRole = this.rankedRoleProbs.get(0).first;
  }
  
  public double getRoleProb(String role) {
	  return this.rankMap.get(role);
  }
  
  public List<Pair<String, Double>> getRankedRoles() {
	  return this.rankedRoleProbs;
  }

  public Counter<String> getProbabilitySRL() {
	  return this.probSRL;
  }
  
  public double getBestRoleProbability() {
	return bestRoleProbability;
  }

  public String getBestRole() {
	  return this.guessRole;
	  //	return this.rankedRoleProbs.get(0).first;
  }

public String role() {
	return this.role;
}

public String getExampleID() {
	return exampleID;
}

public void setExampleID(String exampleID) {
	this.exampleID = exampleID;
}
}