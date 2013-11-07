package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;

public class BioDatum {
  public CoreMap sentence;
  public final String word;
  public final String label;//true label E, O
  public final String role;
  
  FeatureVector features;
  public List<String> getFeatures() {
	return features.getFeatures();
  }

  public void setFeatures(FeatureVector features) {
	this.features = features;
  }
 
  
  /*for event
   * String type = eventNodes.keySet().contains(node) ? "E" : "O";
			BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), type, node, node, exampleID);
			newDatum.features = computeFeatures(sentence, node);
   */
  /*for entity ?? exampleID?
   * String type = (entityNodes.contains(node) && Utils.getArgumentMentionRelation(sentence, eventNode, node) != RelationType.NONE) ? "E" : "O";
	BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), type, node, eventNode, Utils.getArgumentMentionRelation(sentence, eventNode, node).toString());
	newDatum.features = computeFeatures(sentence, node, eventNode);
   */
  /*for event-event
   * String type = Utils.getEventEventRelation(ex.gold, event1.getTreeNode(), event2.getTreeNode()).toString();
	BioDatum newDatum = new BioDatum(null, Utils.getText(event1.getTreeNode()) + "-" + Utils.getText(event2.getTreeNode()), type, event1, event2);
   */
  
  public String exampleID; //sentence ID?
  public String guessLabel; //predicted label, E, O (E is yes, O is not, for both entity and event)
  public String guessRole;
  public int bestRoleIndex;
  public double bestRoleProbability;
  public Tree entityNode;
  public Tree eventNode;
  public EventMention event1, event2;
  IndexedWord entityHead, eventHead;
  public int event1_index, event2_index;//for event-event relation
  public int eventId;//for entity prediction
  double probEntity; //probability of being an entity (E, O)
  double probEvent; //probability of being an event (E, O)
  //double[] probSRL = new double[ArgumentRelation.getSemanticRoles().size()];
  Counter<String> probSRL;
  List<Pair<String, Double>> rankedRoleProbs = new ArrayList<Pair<String, Double>>();
  HashMap<String, Double> rankMap = new HashMap<String, Double>();
  public HashMap<String, Double> rankRelation = new HashMap<String, Double>();
  
  public BioDatum(CoreMap sentence, String word, String label, Tree entityNode, Tree eventNode, String exampleID) {
	this.setSentence(sentence);
    this.word = word;
    this.label = label;
    this.entityNode = entityNode;
    this.eventNode = eventNode;
    this.role = null;
    this.setExampleID(exampleID);
  }
  
  public BioDatum(CoreMap sentence, String word, String label, Tree entityNode, Tree eventNode, String role, String exampleID) {
		this.setSentence(sentence);
	    this.word = word;
	    this.label = label;
	    this.entityNode = entityNode;
	    this.eventNode = eventNode;
	    this.role = role;
	    this.setExampleID(exampleID);
  }
  
  public BioDatum(CoreMap sentence, String word, String label,
		EventMention event1, EventMention event2) {
		this.setSentence(sentence);
	    this.word = word;
	    this.label = label;
	    this.event1 = event1;
	    this.event2 = event2;
	    this.role = null;
}

  public void setPredictedLabel(String predictedLabel) {
	this.guessLabel = predictedLabel;	
  }
  
  public void setEventIndex(int event1_index, int event2_index) {
		this.event1_index = event1_index;
		this.event2_index = event2_index;
	  }

  public void setRankRelation(HashMap<String, Double> rankRelation) {
	this.rankRelation = rankRelation;	
  }
  
  public Double getRelationProb(String relation) {
		return this.rankRelation.get(relation);
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
  
  public void setEventProbability(double prob) {
	  probEvent = prob;
  }
  
  public double getEventProbability() {
	  return probEvent;
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

public CoreMap getSentence() {
	return sentence;
}

void setSentence(CoreMap sentence) {
	this.sentence = sentence;
}
}