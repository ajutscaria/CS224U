package edu.stanford.nlp.bioprocess;

import java.util.*;


import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import fig.basic.LogInfo;

public class Scorer {
/*
  public static Triple<Double, Double, Double> score(List<edu.stanford.nlp.bioprocess.Datum> data) {
    int tp = 0, fp = 0, fn = 0;

    for (Datum d:data) {
    	if(d.label.equals("E")) {
    		if(d.guessLabel.equals("E"))
    			tp++;
    		else
    			fn++;
    	}
    	if(d.label.equals("O")) {
    		if(d.guessLabel.equals("E"))
    			fp++;
    	}
    }
    double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
    double f= 2 * precision * recall / (precision + recall);
    
    //LogInfo.logs("precision = "+precision);
    //LogInfo.logs("recall = "+recall);
    //LogInfo.logs("F1 = "+f);
    
    return new Triple<Double, Double, Double>(precision, recall, f);
  }*/
  
  public static Triple<Double, Double, Double> score(List<BioDatum> dataset) {
	    int tp = 0, fp = 0, fn = 0;

	    for (BioDatum d:dataset) {
	    	if(d.label().equals("E")) {
	    		if(d.predictedLabel().equals("E"))
	    			tp++;
	    		else
	    			fn++;
	    	}
	    	if(d.label().equals("O")) {
	    		if(d.predictedLabel().equals("E"))
	    			fp++;
	    	}
	    }
	    double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
	    double f= 2 * precision * recall / (precision + recall);
	    
	    //LogInfo.logs("precision = "+precision);
	    //LogInfo.logs("recall = "+recall);
	    //LogInfo.logs("F1 = "+f);
	    
	    return new Triple<Double, Double, Double>(precision, recall, f);
	  }
  
  public static Triple<Double, Double, Double> scoreEvents(List<Example> test, List<BioDatum> predictedEvents) {
	  
	  IdentityHashSet<Tree> actual = findActualEvents(test), predicted = findPredictedEvents(predictedEvents);
		int tp = 0, fp = 0, fn = 0;
		for(Tree p:actual) {
			if(predicted.contains(p)) {
				tp++;
				//LogInfo.logs("Correct - " + p) ;
			}
			else {
				fn++;
				//LogInfo.logs("Not predicted - " + p);
			}
		}
		for(Tree p:predicted) {
			if(!actual.contains(p)) {
				fp++;
				//LogInfo.logs("Extra - " + p);
			}
		}
		
		LogInfo.logs("tp fn fp " + tp + ":" + fn + ":" + fp);
		
		 double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
		    double f= 2 * precision * recall / (precision + recall);
		    
		    return new Triple<Double, Double, Double>(precision, recall, f);
	  }

  public static Triple<Double, Double, Double> scoreEntities(List<Example> test, List<BioDatum> predictedEntities) {
	IdentityHashMap<Pair<Tree, Tree>, Integer> actual = findActualEventEntityPairs(test), predicted = findPredictedEventEntityPairs(predictedEntities);
	int tp = 0, fp = 0, fn = 0;
	for(Pair<Tree, Tree> p:actual.keySet()) {
		if(checkContainment(predicted.keySet(),p)) {
			tp++;
			//LogInfo.logs("Correct - " + p.first + ":" + p.second);
		}
		else {
			fn++;
			//LogInfo.logs("Not predicted - " + p.first + ":" + p.second);
		}
	}
	for(Pair<Tree, Tree> p:predicted.keySet()) {
		if(!checkContainment(actual.keySet(),p)) {
			fp++;
			//LogInfo.logs("Extra - " + p.first + ":" + p.second);
		}
	}
	
	LogInfo.logs("tp fn fp " + tp + ":" + fn + ":" + fp);
	
	 double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
	    double f= 2 * precision * recall / (precision + recall);
	    
	    return new Triple<Double, Double, Double>(precision, recall, f);
  }
  
  public static Triple<Double, Double, Double> scoreSRL(List<Example> test, List<BioDatum> predictedEntities) {
	IdentityHashMap<Pair<Tree, Tree>, String> actual = findActualEventEntityRelationPairs(test), predicted = findPredictedEventEntityRelationPairs(predictedEntities);
	int tp = 0, fp = 0, fn = 0;
	for(Pair<Tree, Tree> p : actual.keySet()) {
		Pair<Tree, Tree> pairObjPredicted = returnTreePairIfExists(predicted.keySet(),p);
		if( pairObjPredicted != null && predicted.get(pairObjPredicted).equals(actual.get(p))) {
			tp++;
			//LogInfo.logs("Correct - " + p.first + ":" + p.second);
		}
		else {
			fn++;
			//LogInfo.logs("Not predicted - " + p.first + ":" + p.second);
		}
	}
	for(Pair<Tree, Tree> p:predicted.keySet()) {
		Pair<Tree, Tree> pairObjActual = returnTreePairIfExists(actual.keySet(),p);
		if(pairObjActual == null || !predicted.get(p).equals(actual.get(pairObjActual))) {
			fp++;
			//LogInfo.logs("Extra - " + p.first + ":" + p.second);
		}
	}
	
	LogInfo.logs("tp fn fp " + tp + ":" + fn + ":" + fp);
	
	 double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
	    double f= 2 * precision * recall / (precision + recall);
	    
	    return new Triple<Double, Double, Double>(precision, recall, f);
  }
  
  public static Triple<Double, Double, Double> scoreEventRelations(List<Example> test, List<BioDatum> predictedRelations) {
	IdentityHashMap<Pair<Tree, Tree>, String> actual = findActualEventEventRelationPairs(test), predicted = findPredictedEventEventRelationPairs(predictedRelations);
	int tp = 0, fp = 0, fn = 0;
	for(Pair<Tree, Tree> p : actual.keySet()) {
		Pair<Tree, Tree> pairObjPredicted = returnTreePairIfExists(predicted.keySet(),p);
		if( pairObjPredicted != null && predicted.get(pairObjPredicted).equals(actual.get(p))) {
			tp++;
			//LogInfo.logs("Correct - " + p.first + ":" + p.second + "-->" + actual.get(p));
		}
		else {
			fn++;
			//LogInfo.logs("Not predicted - " + p.first + ":" + p.second+ "-->" + actual.get(p));
		}
	}
	for(Pair<Tree, Tree> p:predicted.keySet()) {
		Pair<Tree, Tree> pairObjActual = returnTreePairIfExists(actual.keySet(),p);
		if(pairObjActual == null || !predicted.get(p).equals(actual.get(pairObjActual))) {
			fp++;
			//LogInfo.logs("Extra - " + p.first + ":" + p.second+ "-->" + predicted.get(p));
		}
	}
	
	LogInfo.logs("tp fn fp " + tp + ":" + fn + ":" + fp);
	
	 double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
	    double f= 2 * precision * recall / (precision + recall);
	    
	    return new Triple<Double, Double, Double>(precision, recall, f);
  }
  
  private static IdentityHashMap<Pair<Tree, Tree>, String> findPredictedEventEntityRelationPairs(
		List<BioDatum> predicted) {
	  IdentityHashMap<Pair<Tree, Tree>, String> map = new IdentityHashMap<Pair<Tree, Tree>, String> ();
	  //LogInfo.begin_track("Gold event-entity relation");
	  for(BioDatum d:predicted) {
		 if(!d.guessRole.equals(RelationType.NONE.toString()))
			 map.put(new Pair<Tree,Tree>(d.eventNode, d.entityNode), d.guessRole);
	  }
	  //LogInfo.end_track();
	  return map;
}

private static IdentityHashMap<Pair<Tree, Tree>, String> findActualEventEntityRelationPairs(
		List<Example> test) {
	IdentityHashMap<Pair<Tree, Tree>, String> map = new IdentityHashMap<Pair<Tree, Tree>, String> ();
	  //LogInfo.begin_track("Gold event-entity");
	  for(Example ex:test) {
		  for(EventMention em:ex.gold.get(EventMentionsAnnotation.class)) {
			  for(ArgumentRelation rel:em.getArguments()) {
				  if(rel.mention instanceof EntityMention && rel.type != RelationType.NONE) {
					  map.put(new Pair<Tree, Tree>(em.getTreeNode(), rel.mention.getTreeNode()), rel.type.toString());
					  //LogInfo.logs(em.getTreeNode() + ":"+ rel.mention.getTreeNode());
				  }
			  }
		  }
	  }
	  //LogInfo.end_track();
	  return map;
}

private static IdentityHashMap<Pair<Tree, Tree>, String> findActualEventEventRelationPairs(
		List<Example> test) {
	IdentityHashMap<Pair<Tree, Tree>, String> map = new IdentityHashMap<Pair<Tree, Tree>, String> ();
	  //LogInfo.begin_track("Gold event-entity");
	  for(Example ex:test) {
		  for(EventMention em:ex.gold.get(EventMentionsAnnotation.class)) {
			  for(ArgumentRelation rel:em.getArguments()) {
				  if(rel.mention instanceof EventMention && rel.type != RelationType.NONE) {
					  map.put(new Pair<Tree, Tree>(em.getTreeNode(), rel.mention.getTreeNode()), rel.type.toString());
					  //LogInfo.logs(em.getTreeNode() + ":"+ rel.mention.getTreeNode());
				  }
			  }
		  }
	  }
	  //LogInfo.end_track();
	  return map;
}

private static IdentityHashMap<Pair<Tree, Tree>, String> findPredictedEventEventRelationPairs(
		List<BioDatum> predicted) {
	  IdentityHashMap<Pair<Tree, Tree>, String> map = new IdentityHashMap<Pair<Tree, Tree>, String> ();
	  //LogInfo.begin_track("Gold event-entity relation");
	  for(BioDatum d:predicted) {
		 if(!d.guessLabel.equals(RelationType.NONE.toString()))
			 map.put(new Pair<Tree,Tree>(d.event1.getTreeNode(), d.event2.getTreeNode()), d.guessLabel);
	  }
	  //LogInfo.end_track();
	  return map;
}

public static Pair<Tree, Tree> returnTreePairIfExists(Set<Pair<Tree, Tree>> s, Pair<Tree, Tree> element) {
	  for(Pair<Tree, Tree> keys:s)
		  if(keys.first()==(element.first()) && keys.second()==(element.second()))
			  return keys;
	  return null;
}

public static boolean checkContainment(Set<Pair<Tree, Tree>> s, Pair<Tree, Tree> element) {
	  for(Pair<Tree, Tree> keys:s)
		  if(keys.first()==(element.first()) && keys.second()==(element.second()))
			  return true;
	  return false;
  }
  
  public static IdentityHashMap<Pair<Tree, Tree>, Integer> findActualEventEntityPairs(List<Example> test) {
	  IdentityHashMap<Pair<Tree, Tree>, Integer> map = new IdentityHashMap<Pair<Tree, Tree>, Integer> ();
	  //LogInfo.begin_track("Gold event-entity");
	  for(Example ex:test) {
		  for(EventMention em:ex.gold.get(EventMentionsAnnotation.class)) {
			  for(ArgumentRelation rel:em.getArguments()) {
				  if(rel.mention instanceof EntityMention) {
					  map.put(new Pair(em.getTreeNode(), rel.mention.getTreeNode()), 1);
					  //LogInfo.logs(em.getTreeNode() + ":"+ rel.mention.getTreeNode());
				  }
			  }
		  }
	  }
	  //LogInfo.end_track();
	  return map;
  }
  
  public static IdentityHashMap<Pair<Tree, Tree>, Integer> findPredictedEventEntityPairs(List<BioDatum> predicted) {
	  IdentityHashMap<Pair<Tree, Tree>, Integer> map = new IdentityHashMap<Pair<Tree, Tree>, Integer> ();
	  //LogInfo.begin_track("Gold event-entity");
	  for(BioDatum d:predicted) {
		 if(d.guessLabel.equals("E"))
			 map.put(new Pair<Tree,Tree>(d.eventNode, d.entityNode), 1);
	  }
	  //LogInfo.end_track();
	  return map;
  }
  
  public static IdentityHashSet<Tree> findPredictedEvents(List<BioDatum> predicted) {
	  IdentityHashSet<Tree> set = new IdentityHashSet<Tree>();
	  for(BioDatum d:predicted) {
		 if(d.guessLabel.equals("E"))
			 set.add(d.eventNode);
	  }
	  return set;
  }
  
  public static IdentityHashSet<Tree> findActualEvents(List<Example> test){
	  IdentityHashSet<Tree> set = new IdentityHashSet<Tree>();
	  for(Example ex:test) {
		  //LogInfo.logs("Example " + ex.id);
		  for(EventMention em:ex.gold.get(EventMentionsAnnotation.class)) {
			  //LogInfo.logs(em.getTreeNode() + ":" + em.getTreeNode().getSpan() + "::" + em.getSentence());
			  set.add(em.getTreeNode());
		  }
	  }
	  
	  return set;
  }

  public static Triple<Double, Double, Double> scoreStandaloneEntities(List<Example> test, List<BioDatum> predictedEntities) {
	  IdentityHashSet<Tree> actual = findActualEntities(test), predicted = findPredictedEntities(predictedEntities);
		int tp = 0, fp = 0, fn = 0;
		for(Tree p:actual) {
			if(predicted.contains(p)) {
				tp++;
				//LogInfo.logs("Correct - " + p.first + ":" + p.second);
			}
			else {
				fn++;
				//LogInfo.logs("Not predicted - " + p.first + ":" + p.second);
			}
		}
		for(Tree p:predicted) {
			if(!actual.contains(p)) {
				fp++;
				//LogInfo.logs("Extra - " + p.first + ":" + p.second);
			}
		}
		
		LogInfo.logs("tp fn fp " + tp + ":" + fn + ":" + fp);
		
		 double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
		    double f= 2 * precision * recall / (precision + recall);
		    
		    return new Triple<Double, Double, Double>(precision, recall, f);
	  }
  
  public static IdentityHashSet<Tree> findActualEntities(List<Example> test) {
	  IdentityHashSet<Tree> set = new IdentityHashSet<Tree>();
	  for(Example ex:test) {
		  for(EntityMention em:ex.gold.get(EntityMentionsAnnotation.class)) {
			  set.add(em.getTreeNode());
		  }
	  }
	  
	  return set;
  }
  
  public static IdentityHashSet<Tree> findPredictedEntities(List<BioDatum> predicted) {
	  IdentityHashSet<Tree> set = new IdentityHashSet<Tree>();
	  for(BioDatum d:predicted) {
		 if(d.guessLabel.equals("E"))
			 set.add(d.entityNode);
	  }
	  return set;
  }
}