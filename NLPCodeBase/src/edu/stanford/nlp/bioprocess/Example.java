package edu.stanford.nlp.bioprocess;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import fig.basic.LogInfo;
import fig.basic.MapUtils;
import fig.basic.Option;
import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;

public class Example implements Serializable {
	/**
	 * 
	 */
	public static class Options {
	    @Option public int verbose = 0; //higher verbose level -> more details
	}
	
    public class Stat{
        int tp;
        int fp;
        int fn;
        public Stat(int tp, int fp, int fn){
        	this.tp = tp;
        	this.fp = fp;
        	this.fn = fn;
        }
    }	
    
	private static final long serialVersionUID = 1L;
	public String id, data;
	public Annotation gold;
	public Annotation prediction;
	public static Options opts = new Options();
	private int tp, fp, fn;
	public List<BioDatum> events;
	public List<BioDatum> entities;
	public List<BioDatum> relations;

	public String getData() {
		return data;
	}

	public void setPrediction(Annotation pred) {
		prediction = pred;
	}
	
	public void printPrediction(){
		for(CoreMap sentence:gold.get(SentencesAnnotation.class)){ 
			if(opts.verbose > 0) {
				LogInfo.logs(sentence);
				LogInfo.logs(sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennString());
				LogInfo.logs(sentence.get(CollapsedCCProcessedDependenciesAnnotation.class));
			}
		}
	}
	
    public Stat scoreEvent(){
    	LogInfo.begin_track("Events:");
    	IdentityHashMap<Tree, Double> actual = findEvents(gold);
    	IdentityHashMap<Tree, Double> predicted = findEvents(prediction);
    	tp = 0; fp = 0; fn = 0;
    	
    	for(EventMention em:prediction.get(EventMentionsAnnotation.class)) {
			/*if(em.prob >= 0.5 && actual.containsKey(em.getTreeNode())){
				tp++;
				LogInfo.logs("tp: "+
						   String.format("%-30s Predicted=%s", em.getTreeNode(), "E"));
			}else if(em.prob >= 0.5 && !actual.containsKey(em.getTreeNode())){
				fp++;
				LogInfo.logs("fp: "+
						   String.format("%-30s Predicted=%s", em.getTreeNode(), "E"));
			}else if(em.prob < 0.5 && actual.containsKey(em.getTreeNode())){
				fn++;
				LogInfo.logs("fn: "+
						   String.format("%-30s Predicted=%s", em.getTreeNode(), "O"));
			}*/
			
			if(actual.containsKey(em.getTreeNode())){
				tp++;
				LogInfo.logs("tp: "+
						   String.format("%-30s Predicted=%s", em.getTreeNode(), "E"));
			}else{// if(!actual.containsKey(em.getTreeNode())){
				fp++;
				LogInfo.logs("fp: "+
						   String.format("%-30s Predicted=%s", em.getTreeNode(), "E"));
			}
		}
    	for(EventMention em:gold.get(EventMentionsAnnotation.class)) {
    		if(!predicted.containsKey(em.getTreeNode())){
				fn++;
				LogInfo.logs("fn: "+
						   String.format("%-30s Predicted=%s", em.getTreeNode(), "O"));
			}
    	}
    	/*for(Tree p:actual.keySet()) {
			if(predicted.containsKey(p)) {
				tp++;
				LogInfo.logs("tp: "+
				   String.format("%-30s Predicted=%s", p, "E"));
			}
			else {
				fn++;
				LogInfo.logs("fn: "+
				   String.format("%-30s Predicted=%s", p, "O"));
			}
		}
		for(Tree p:predicted.keySet()) {
			if(!actual.containsKey(p)) {
				fp++;
				LogInfo.logs("fp: "+
						   String.format("%-30s Predicted=%s", p, "E"));
			}
		}*/
		double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
		double f= 2 * precision * recall / (precision + recall);
		LogInfo.logs("Event p/r/f=%.4f, %.4f, %.4f",precision, recall, f);
    	LogInfo.end_track();
		//return new Triple<Integer, Integer, Integer>(tp, fp, fn);
    	return new Stat(tp, fp, fn);
	}
    
    public Stat scoreEntity(){
    	LogInfo.begin_track("Entities:");
    	tp = 0; fp = 0; fn = 0;
    	IdentityHashMap<Pair<Tree, Tree>, Double> actual = findEventEntityPairs(gold),
				predicted = findEventEntityPairs(prediction);
    	
    	for(EventMention em:prediction.get(EventMentionsAnnotation.class)) {
			for(ArgumentRelation rel:em.getArguments()) {
				if(rel.mention instanceof EntityMention) {
					if(checkContainment(actual.keySet(),new Pair(em.getTreeNode(), rel.mention.getTreeNode()))) {
						tp++;
						LogInfo.logs("tp: "+
						  String.format("Event=%-15s Entity=%s, Predicted=%s", em.getTreeNode(), rel.mention.getTreeNode(), "E"));
					}
					else{
						fp++;
						LogInfo.logs("fp: "+
								  String.format("Event=%-15s Entity=%s, Predicted=%s", em.getTreeNode(), rel.mention.getTreeNode(), "E"));
				    }
				}
			}
		}
    	
    	for(EventMention em:gold.get(EventMentionsAnnotation.class)) {
			for(ArgumentRelation rel:em.getArguments()) {
				if(rel.mention instanceof EntityMention) {
					if(!checkContainment(predicted.keySet(),new Pair(em.getTreeNode(), rel.mention.getTreeNode()))) {
						fn++;
						LogInfo.logs("fn: "+
						  String.format("Event=%-15s Entity=%s, Predicted=%s", em.getTreeNode(), rel.mention.getTreeNode(), "O"));
					}
				}
			}
		}
		/*for(Pair<Tree, Tree> p:actual.keySet()) {
			if(checkContainment(predicted.keySet(),p)) {
				tp++;
				LogInfo.logs("tp: "+
				  String.format("Event=%-15s Entity=%s, Predicted=%s", p.first, p.second, "E"));
			}
			else {
				fn++;
				LogInfo.logs("fn: "+
						  String.format("Event=%-15s Entity=%s, Predicted=%s", p.first, p.second, "O"));
			}
		}
		for(Pair<Tree, Tree> p:predicted.keySet()) {
			if(!checkContainment(actual.keySet(),p)) {
				fp++;
				LogInfo.logs("fp: "+
						  String.format("Event=%-15s Entity=%s, Predicted=%s", p.first, p.second, "E"));
			}
		}
		*/
    	double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
		double f= 2 * precision * recall / (precision + recall);
		LogInfo.logs("Entity p/r/f=%.4f, %.4f, %.4f",precision, recall, f);
    	LogInfo.end_track();
		//return new Triple<Integer, Integer, Integer>(tp, fp, fn);
    	return new Stat(tp, fp, fn);
	}

    public Stat scoreRelation(){
    	LogInfo.begin_track("Event-Event Relations:");
    	tp = 0; fp = 0; fn = 0; 
    	int fn_null = 0, fn_diff = 0;
    	
    	IdentityHashMap<Pair<Tree, Tree>, String> actual = findEventEventRelationPairs(gold), 
				predicted = findPredictedEventEventRelationPairs(prediction);

    	for(EventMention em:prediction.get(EventMentionsAnnotation.class)){
			//System.out.println(em.getTreeNode().toString());
			//System.out.println("relations:");
			for(ArgumentRelation rel:em.getArguments()) {
				if(rel.mention instanceof EventMention && rel.type != RelationType.NONE) {
					Pair<Tree, Tree> p = new Pair<Tree, Tree>(em.getTreeNode(), rel.mention.getTreeNode());
					String type = returnTypeIfExists(predicted, p);
					Pair<Tree, Tree> pairObjPredicted = returnTreePairIfExists(predicted.keySet(),p);
					Pair<Tree, Tree> pairObjActual = returnTreePairIfExists(actual.keySet(),p);
					
					if(pairObjActual == null || !predicted.get(pairObjPredicted).equals(actual.get(pairObjActual))) {
						fp++;
						LogInfo.logs("fp: "+
								  String.format("Event1=%-15s Event2=%-15s, Predicted=%s", p.first, p.second, type));
					}else{
						tp++;
						LogInfo.logs("tp: "+
								  String.format("Event1=%-15s Event2=%-15s, Predicted=%s", p.first, p.second, type));
					}
				}
			}
		}
    	
    	List<EventMention> list = gold.get(EventMentionsAnnotation.class);
		List<EventMention> alreadyConsidered = new ArrayList<EventMention>();
		for(EventMention event1:list) {
			alreadyConsidered.add(event1);
			for(EventMention event2: list) {
				if(!alreadyConsidered.contains(event2)) {
					String type = Utils.getEventEventRelation(gold, event1.getTreeNode(), event2.getTreeNode()).toString();
					if(!type.equals("NONE")){
						Pair<Tree, Tree> p = new Pair<Tree, Tree>(event1.getTreeNode(), event2.getTreeNode());
						Pair<Tree, Tree> pairObjActual = returnTreePairIfExists(actual.keySet(),p);
						Pair<Tree, Tree> pairObjPredicted = returnTreePairIfExists(predicted.keySet(),p);
						if( pairObjPredicted == null || !predicted.get(pairObjPredicted).equals(actual.get(pairObjActual))) {
							fn++;
							LogInfo.logs("fn: "+
									  String.format("Event1=%-15s Event2=%-15s, Predicted=%s", p.first, p.second, actual.get(pairObjActual)));
						}
					}
				}
			}

		}
    	/*
		for(Pair<Tree, Tree> p : actual.keySet()) {
			//System.out.println(p.first+","+ p.second+"->"+actual.get(p));
			Pair<Tree, Tree> pairObjPredicted = returnTreePairIfExists(predicted.keySet(),p);
			if( pairObjPredicted != null && predicted.get(pairObjPredicted).equals(actual.get(p))) {
				tp++;
				LogInfo.logs("tp: "+
						  String.format("Event1=%-15s Event2=%-15s, Predicted=%s", p.first, p.second, actual.get(p)));
			}
			else {
				if(pairObjPredicted == null){
					fn_null++;
				}
				else if(!predicted.get(pairObjPredicted).equals(actual.get(p)))fn_diff++;
				fn++;
				LogInfo.logs("fn: "+
						  String.format("Event1=%-15s Event2=%-15s, Predicted=%s", p.first, p.second, actual.get(p)));
			}
		}
		//System.out.println("Predicted:");
		for(Pair<Tree, Tree> p:predicted.keySet()) {
			Pair<Tree, Tree> pairObjActual = returnTreePairIfExists(actual.keySet(),p);
			if(pairObjActual == null || !predicted.get(p).equals(actual.get(pairObjActual))) {
				fp++;
				LogInfo.logs("fp: "+
						  String.format("Event1=%-15s Event2=%-15s, Predicted=%s", p.first, p.second, predicted.get(p)));
			}
		}
    	*/
    	double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
    	double f;
    	if(precision + recall < 0.000001)
			f = 0;
		else
			f= 2 * precision * recall / (precision + recall);
		LogInfo.logs("Relation p/r/f=%.4f, %.4f, %.4f",precision, recall, f);
    	LogInfo.end_track();
	    //return new Triple<Integer, Integer, Integer>(tp, fp, fn);
    	return new Stat(tp, fp, fn);
    }
    
    private IdentityHashMap<Pair<Tree, Tree>, String> findEventEventRelationPairs(Annotation ann) {
		IdentityHashMap<Pair<Tree, Tree>, String> map = new IdentityHashMap<Pair<Tree, Tree>, String> ();
		//LogInfo.begin_track("Gold event-entity");

		List<EventMention> list = ann.get(EventMentionsAnnotation.class);
		List<EventMention> alreadyConsidered = new ArrayList<EventMention>();
		for(EventMention event1:list) {
			alreadyConsidered.add(event1);
			for(EventMention event2: list) {
				if(!alreadyConsidered.contains(event2)) {
					String type = Utils.getEventEventRelation(ann, event1.getTreeNode(), event2.getTreeNode()).toString();
					if(!type.equals("NONE")){
						map.put(new Pair<Tree, Tree>(event1.getTreeNode(), event2.getTreeNode()), type.toString());
					}
				}
			}

		}
		//LogInfo.end_track();
		return map;
	}
    
    private IdentityHashMap<Pair<Tree, Tree>, String> findPredictedEventEventRelationPairs(Annotation ann) {
		IdentityHashMap<Pair<Tree, Tree>, String> map = new IdentityHashMap<Pair<Tree, Tree>, String> ();
		//LogInfo.begin_track("Gold event-entity");
		for(EventMention em:ann.get(EventMentionsAnnotation.class)){
			//System.out.println(em.getTreeNode().toString());
			//System.out.println("relations:");
			for(ArgumentRelation rel:em.getArguments()) {
				if(rel.mention instanceof EventMention && rel.type != RelationType.NONE) {
					map.put(new Pair<Tree, Tree>(em.getTreeNode(), rel.mention.getTreeNode()), rel.type.toString());
				}
			}
		}
		//LogInfo.end_track();
		return map;
	}
    
    
    public static String returnTypeIfExists(IdentityHashMap<Pair<Tree, Tree>, String> s, Pair<Tree, Tree> element) {
		for(Pair<Tree, Tree> keys:s.keySet())
			if(keys.first()==(element.first()) && keys.second()==(element.second()))
				return s.get(keys);
		return null;
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
    
    public static IdentityHashMap<Pair<Tree, Tree>, Double> findEventEntityPairs(Annotation ann) {
		IdentityHashMap<Pair<Tree, Tree>, Double> map = new IdentityHashMap<Pair<Tree, Tree>, Double> ();
		//LogInfo.begin_track("Gold event-entity");
		
		for(EventMention em:ann.get(EventMentionsAnnotation.class)) {
			for(ArgumentRelation rel:em.getArguments()) {
				if(rel.mention instanceof EntityMention) {
					map.put(new Pair(em.getTreeNode(), rel.mention.getTreeNode()), rel.getProb());
					//LogInfo.logs(em.getTreeNode() + ":"+ rel.mention.getTreeNode());
				}
			}
		}
		
		//LogInfo.end_track();
		return map;
	}
    
    public IdentityHashMap<Tree, Double> findEvents(Annotation ann){//linkedhashmap
		IdentityHashMap<Tree, Double> set = new IdentityHashMap<Tree, Double>();
			//LogInfo.logs("Example " + ex.id);
		for(EventMention em:ann.get(EventMentionsAnnotation.class)) {
			//LogInfo.logs(em.getTreeNode() + ":" + em.getTreeNode().getSpan() + "::" + em.getSentence());
			set.put(em.getTreeNode(), em.prob);
		}
		
		return set;
	}
}
