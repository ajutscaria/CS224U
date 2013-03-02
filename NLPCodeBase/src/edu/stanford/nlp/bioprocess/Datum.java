package edu.stanford.nlp.bioprocess;

import java.util.*;

import edu.stanford.nlp.trees.Tree;

public class Datum {

  public final String word;
  public final String label;
  public List<String> features;
  public String guessLabel;
  Tree entityNode, eventNode;
  double probEntity;
  
  public Datum(String word, String label, Tree entityNode, Tree eventNode) {
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
}