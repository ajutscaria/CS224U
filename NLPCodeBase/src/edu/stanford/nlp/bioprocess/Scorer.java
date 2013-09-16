package edu.stanford.nlp.bioprocess;

import java.util.*;


import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.IntCounter;
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
		
		//LogInfo.logs("tp fn fp " + tp + ":" + fn + ":" + fp);
		
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
	
	//LogInfo.logs("tp fn fp " + tp + ":" + fn + ":" + fp);
	
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
	
	//LogInfo.logs("tp fn fp " + tp + ":" + fn + ":" + fp);
	
	 double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
	    double f= 2 * precision * recall / (precision + recall);
	    
	    return new Triple<Double, Double, Double>(precision, recall, f);
  }
  
  
  public static Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> scoreEventRelations(List<BioDatum> predictedRelations) {
	IntCounter<String> truePos = new IntCounter<String>(), falsePos = new IntCounter<String>(), falseNeg = new IntCounter<String>();
	int tp = 0, fp = 0, fn = 0;
	for(BioDatum d:predictedRelations) {
		if(d.predictedLabel().equals(d.label()) && d.predictedLabel().equals("NONE")) {
			truePos.incrementCount(d.label);
		}
		else if(d.predictedLabel().equals(d.label)) {
			tp++;
			truePos.incrementCount(d.label);
		}
		else if(d.label().equals("NONE") && !d.predictedLabel().equals("NONE")){
			fp++;
			falsePos.incrementCount(d.predictedLabel());
			falseNeg.incrementCount(d.label);
		}
		else if(!d.label().equals("NONE")){
			fn++;
			if(!d.predictedLabel().equals("NONE"))
				fp++;
			falseNeg.incrementCount(d.label);
			falsePos.incrementCount(d.predictedLabel());
		}
		//if(d.label.equals("SameEvent") && d.predictedLabel().equals("NONE")) {
		//	LogInfo.logs(d.event1.getTreeNode() + ":" + d.event2.getTreeNode() + "-->" + d.getExampleID());
		//}
	}
	
	HashMap<String, Double> macroPrecision = new HashMap<String, Double>(), macroRecall = new HashMap<String, Double>(), macroF1 = new HashMap<String, Double>();
	
	List<String> nonNONERelations = ArgumentRelation.getEventRelations();
	nonNONERelations.remove("NONE");
	
	for(String key : nonNONERelations) {
		macroPrecision.put(key, 0.0);
		macroRecall.put(key, 0.0);
		macroF1.put(key, 0.0);
	}
	
	for(String key : nonNONERelations) {
		double precision = (truePos.getCount(key) + falsePos.getCount(key)) == 0 ? 1 :  (double) truePos.getCount(key) / (truePos.getCount(key) + falsePos.getCount(key));
		double recall = (truePos.getCount(key) + falseNeg.getCount(key)) == 0 ? 0 :  (double) truePos.getCount(key) / (truePos.getCount(key) + falseNeg.getCount(key));
		macroPrecision.put(key, precision);
		macroRecall.put(key, recall);
		macroF1.put(key, precision == 0 && recall == 0 ? 0 : (double) 2 * precision * recall / (precision + recall));
	}
	double macroPrec = 0, macroRec = 0, macroF = 0;
	
	//LogInfo.logs(macroPrecision);
	//LogInfo.logs(macroRecall);
	//LogInfo.logs(macroF1);
	
	for(String key : nonNONERelations) {
		macroPrec += macroPrecision.get(key);
		macroRec += macroRecall.get(key);
		macroF += macroF1.get(key);
	}
	
	macroPrec = macroPrec / nonNONERelations.size();
	macroRec = macroRec / nonNONERelations.size();
	macroF = macroF / nonNONERelations.size();
	
	//LogInfo.logs("tp fn fp " + tp + ":" + fn + ":" + fp);
	
	double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
	double f= 2 * precision * recall / (precision + recall);
	    
	return new Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>>(new Triple<Double, Double, Double>(precision, recall, f), new Triple<Double, Double, Double>(macroPrec, macroRec, macroF));
  }
  
  public static Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> scoreEventRelationsCollapsed(List<BioDatum> predictedRelations) {
		IntCounter<String> truePos = new IntCounter<String>(), falsePos = new IntCounter<String>(), falseNeg = new IntCounter<String>();
		int tp = 0, fp = 0, fn = 0;
		List<String> PreviousEventCluster = new ArrayList<String>();
		PreviousEventCluster.add("PreviousEvent");
		PreviousEventCluster.add("Causes");
		PreviousEventCluster.add("Enables");
		
		List<String> NextEventCluster = new ArrayList<String>();
		NextEventCluster.add("NextEvent");
		NextEventCluster.add("Caused");
		NextEventCluster.add("Enabled");
		
		for(BioDatum d:predictedRelations) {
			if(d.predictedLabel().equals(d.label()) && d.predictedLabel().equals("NONE")) {
				truePos.incrementCount(d.label);
			}
			else if(d.predictedLabel().equals(d.label) ||
					(PreviousEventCluster.contains(d.predictedLabel()) && PreviousEventCluster.contains(d.label)) ||
					(NextEventCluster.contains(d.predictedLabel()) && NextEventCluster.contains(d.label))) {
				tp++;
				truePos.incrementCount(d.label);
			}
			else if(d.label().equals("NONE") && !d.predictedLabel().equals("NONE")){
				fp++;
				falsePos.incrementCount(d.predictedLabel());
				falseNeg.incrementCount(d.label);
			}
			else{
				fn++;
				if(!d.predictedLabel().equals("NONE"))
					fp++;
				falseNeg.incrementCount(d.label);
				falsePos.incrementCount(d.predictedLabel());
			}
			//if(d.label.equals("SameEvent") && d.predictedLabel().equals("NONE")) {
			//	LogInfo.logs(d.event1.getTreeNode() + ":" + d.event2.getTreeNode() + "-->" + d.getExampleID());
			//}
		}
		
		HashMap<String, Double> macroPrecision = new HashMap<String, Double>(), macroRecall = new HashMap<String, Double>(), macroF1 = new HashMap<String, Double>();
		
		List<String> nonNONERelations = ArgumentRelation.getEventRelations();
		nonNONERelations.remove("NONE");
		
		for(String key : nonNONERelations) {
			macroPrecision.put(key, 0.0);
			macroRecall.put(key, 0.0);
			macroF1.put(key, 0.0);
		}
		
		for(String key : nonNONERelations) {
			double precision = (truePos.getCount(key) + falsePos.getCount(key)) == 0 ? 1 :  (double) truePos.getCount(key) / (truePos.getCount(key) + falsePos.getCount(key));
			double recall = (truePos.getCount(key) + falseNeg.getCount(key)) == 0 ? 0 :  (double) truePos.getCount(key) / (truePos.getCount(key) + falseNeg.getCount(key));
			macroPrecision.put(key, precision);
			macroRecall.put(key, recall);
			macroF1.put(key, precision == 0 && recall == 0 ? 0 : (double) 2 * precision * recall / (precision + recall));
		}
		double macroPrec = 0, macroRec = 0, macroF = 0;
		
		//LogInfo.logs(macroPrecision);
		//LogInfo.logs(macroRecall);
		//LogInfo.logs(macroF1);
		
		for(String key : nonNONERelations) {
			macroPrec += macroPrecision.get(key);
			macroRec += macroRecall.get(key);
			macroF += macroF1.get(key);
		}
		
		macroPrec = macroPrec / nonNONERelations.size();
		macroRec = macroRec / nonNONERelations.size();
		macroF = macroF / nonNONERelations.size();
		
		//LogInfo.logs("tp fn fp " + tp + ":" + fn + ":" + fp);
		
		double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
		double f= 2 * precision * recall / (precision + recall);
		    
		return new Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>>(new Triple<Double, Double, Double>(precision, recall, f), new Triple<Double, Double, Double>(macroPrec, macroRec, macroF));
	  }
  
  public static Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> scoreEventRelationsStructure(List<BioDatum> predictedRelations) {
		IntCounter<String> truePos = new IntCounter<String>(), falsePos = new IntCounter<String>(), falseNeg = new IntCounter<String>();
		int tp = 0, fp = 0, fn = 0;
		List<String> PreviousEventCluster = new ArrayList<String>();
		PreviousEventCluster.add("PreviousEvent");
		PreviousEventCluster.add("Causes");
		PreviousEventCluster.add("Enables");
		
		List<String> NextEventCluster = new ArrayList<String>();
		PreviousEventCluster.add("NextEvent");
		PreviousEventCluster.add("Caused");
		PreviousEventCluster.add("Enabled");
		
		for(BioDatum d:predictedRelations) {
			if(d.predictedLabel().equals(d.label()) && d.predictedLabel().equals("NONE")) {
				truePos.incrementCount(d.label);
			}
			else if(d.predictedLabel().equals(d.label) ||
					(!d.predictedLabel().equals("NONE") && !d.label.equals("NONE"))) {
				tp++;
				truePos.incrementCount(d.label);
			}
			else if(d.label().equals("NONE") && !d.predictedLabel().equals("NONE")){
				fp++;
				falsePos.incrementCount(d.predictedLabel());
				falseNeg.incrementCount(d.label);
			}
			else if(!d.label().equals("NONE")){
				fn++;
				if(!d.predictedLabel().equals("NONE"))
					fp++;
				falseNeg.incrementCount(d.label);
				falsePos.incrementCount(d.predictedLabel());
			}
			//if(d.label.equals("SameEvent") && d.predictedLabel().equals("NONE")) {
			//	LogInfo.logs(d.event1.getTreeNode() + ":" + d.event2.getTreeNode() + "-->" + d.getExampleID());
			//}
		}
		
		HashMap<String, Double> macroPrecision = new HashMap<String, Double>(), macroRecall = new HashMap<String, Double>(), macroF1 = new HashMap<String, Double>();
		
		List<String> nonNONERelations = ArgumentRelation.getEventRelations();
		nonNONERelations.remove("NONE");
		
		for(String key : nonNONERelations) {
			macroPrecision.put(key, 0.0);
			macroRecall.put(key, 0.0);
			macroF1.put(key, 0.0);
		}
		
		for(String key : nonNONERelations) {
			double precision = (truePos.getCount(key) + falsePos.getCount(key)) == 0 ? 1 :  (double) truePos.getCount(key) / (truePos.getCount(key) + falsePos.getCount(key));
			double recall = (truePos.getCount(key) + falseNeg.getCount(key)) == 0 ? 0 :  (double) truePos.getCount(key) / (truePos.getCount(key) + falseNeg.getCount(key));
			macroPrecision.put(key, precision);
			macroRecall.put(key, recall);
			macroF1.put(key, precision == 0 && recall == 0 ? 0 : (double) 2 * precision * recall / (precision + recall));
		}
		double macroPrec = 0, macroRec = 0, macroF = 0;
		
		//LogInfo.logs(macroPrecision);
		//LogInfo.logs(macroRecall);
		//LogInfo.logs(macroF1);
		
		for(String key : nonNONERelations) {
			macroPrec += macroPrecision.get(key);
			macroRec += macroRecall.get(key);
			macroF += macroF1.get(key);
		}
		
		macroPrec = macroPrec / nonNONERelations.size();
		macroRec = macroRec / nonNONERelations.size();
		macroF = macroF / nonNONERelations.size();
		
		//LogInfo.logs("tp fn fp " + tp + ":" + fn + ":" + fp);
		
		double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
		double f= 2 * precision * recall / (precision + recall);
		    
		return new Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>>(new Triple<Double, Double, Double>(precision, recall, f), new Triple<Double, Double, Double>(macroPrec, macroRec, macroF));
	  }
  
  public static Triple<Double, Double, Double> scoreEventRelationsPipeline(List<Example> examples, List<BioDatum> predictedRelations) {
		
		int tp = 0, fp = 0, fn = 0;

		for(Example example:examples) {
			LogInfo.logs("Example "+ example.id);
			List<EventMention> mentionsPredicted = new ArrayList<EventMention>();
			for(BioDatum dat:predictedRelations) {
				if(dat.getExampleID().equals(example.id)) {
					if(!mentionsPredicted.contains(dat.event1))
						mentionsPredicted.add(dat.event1);
					else if(!mentionsPredicted.contains(dat.event2))
						mentionsPredicted.add(dat.event2);
					
					if(dat.predictedLabel().equals(dat.label()) && dat.predictedLabel().equals("NONE")) {
						//Nothing
					}
					else if(dat.predictedLabel().equals(dat.label)) {
						tp++;
						LogInfo.logs(String.format("\t%s %-10s : %-10s - %-10s Gold:  %s Predicted: %s", example.id,  "Correct", 
									Utils.getText(dat.event1.getTreeNode()), Utils.getText(dat.event2.getTreeNode()), dat.label(), dat.predictedLabel()));
					}
					else if(dat.label().equals("NONE") && !dat.predictedLabel().equals("NONE")){
						fp++;
						LogInfo.logs(String.format("\t%s %-10s : %-10s - %-10s Gold:  %s Predicted: %s", example.id,  "Extra", 
								Utils.getText(dat.event1.getTreeNode()), Utils.getText(dat.event2.getTreeNode()), dat.label(), dat.predictedLabel()));
					}
					else if(!dat.label().equals("NONE")){
						fn++;
						LogInfo.logs(String.format("\t%s %-10s : %-10s - %-10s Gold:  %s Predicted: %s", example.id,  "Missed", 
								Utils.getText(dat.event1.getTreeNode()), Utils.getText(dat.event2.getTreeNode()), dat.label(), dat.predictedLabel()));
						if(!dat.predictedLabel().equals("NONE"))
							fp++;
					}
				}
			}
			List<EventMention> mentionsNotPredicted = new ArrayList<EventMention>();
			for(EventMention g:example.gold.get(EventMentionsAnnotation.class)) {
				boolean found = false;
				for(EventMention p:mentionsPredicted) {
					if(p.getTreeNode() == g.getTreeNode())
						found = true;
				}
				if(!found)
					mentionsNotPredicted.add(g);
			}
			List<EventMention> considered = new ArrayList<EventMention>();
			for(EventMention e1:mentionsNotPredicted) {
				for(EventMention e2:example.gold.get(EventMentionsAnnotation.class)) {
					String relation = Utils.getEventEventRelation(example.gold, e1.getTreeNode(), e2.getTreeNode()).toString();
					if(!considered.contains(e2) && e1 != e2 && 
						!relation.equals("NONE")) {
						fn++;
						LogInfo.logs(String.format("\t%s %-10s : %-10s - %-10s Gold:  %s Predicted: %s", example.id,  "Missed", 
								Utils.getText(e1.getTreeNode()), Utils.getText(e2.getTreeNode()), relation, "NONE"));
					}
				}
				considered.add(e1);
			}
			LogInfo.logs("\n\n");
		}
		double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
		double f= 2 * precision * recall / (precision + recall);
		return new Triple<Double, Double, Double>(precision, recall, f);
  }
  /*
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
  }*/
  
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

  public static void updateMatrix(double[][] confusionMatrix, List<BioDatum> predicted, List<String> relations) {
	  for(BioDatum d:predicted) {
		 int actualIndex = relations.indexOf(d.label), predictedIndex = relations.indexOf(d.predictedLabel());
		 confusionMatrix[actualIndex][predictedIndex] += 1;
	  }
  }
}