package edu.stanford.nlp.bioprocess;

import java.util.*;

import edu.stanford.nlp.bioprocess.ArgumentRelation.EventType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.graph.Graph;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.IntPair;
import fig.basic.LogInfo;
import edu.stanford.nlp.util.StringUtils;

public class EventExtendedFeatureFactory extends FeatureExtractor {
	public EventExtendedFeatureFactory(boolean useLexicalFeatures) {
		super(useLexicalFeatures);
	}

	boolean printDebug = false, printAnnotations = false, printFeatures = false;
	boolean addEntityFeatures = true;
	Set<String> nominalizations = Utils.getNominalizedVerbs();
	HashMap<String, String> verbForms = Utils.getVerbForms();
	HashMap<String, Integer> clusters = Utils.loadClustering();
	EventFeatureFactory basicFeatureFactory = new EventFeatureFactory(useLexicalFeatures);
   
	public FeatureVector computeFeatures(CoreMap sentence, Tree event) {
	    return computeFeatures(sentence, event, Utils.getEntityNodesFromSentence(sentence));
    }

    public List<BioDatum> setFeaturesTrain(List<Example> data) {
		List<BioDatum> newData = new ArrayList<BioDatum>();
		
		for (Example ex : data) {
			if(printDebug || printAnnotations) LogInfo.logs("\n-------------------- " + ex.id + "---------------------");
			for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
				IdentityHashMap<Tree, EventType> eventNodes = Utils.getEventNodesFromSentence(sentence);
				if(printDebug) LogInfo.logs(sentence);
				if(printAnnotations) {
					LogInfo.logs("---Events--");
					for(EventMention event: sentence.get(EventMentionsAnnotation.class))
						LogInfo.logs(event.getValue());
					//LogInfo.logs("---Entities--");
					//for(EntityMention entity: sentence.get(EntityMentionsAnnotation.class))
						//LogInfo.logs(entity.getTreeNode() + ":" + entity.getTreeNode().getSpan());
				}
				//for(EventMention event: sentence.get(EventMentionsAnnotation.class)) {
					//if(printDebug) LogInfo.logs("-------Event - " + event.getTreeNode()+ "--------");
					for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
						if(node.isLeaf() || node.value().equals("ROOT") || !node.isPreTerminal() || 
								!(node.value().startsWith("JJR") || node.value().startsWith("JJS") ||node.value().startsWith("NN") || node.value().equals("JJ") || node.value().startsWith("VB")))
							continue;
						
						String type = eventNodes.keySet().contains(node) ? "E" : "O";
						//if(printDebug) LogInfo.logs(type + " : " + node + ":" + node.getSpan());
						BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), type, node, node, ex.id);
						newDatum.features = computeFeatures(sentence, node);
						if(printFeatures) LogInfo.logs(Utils.getText(node) + ":" + newDatum.features.getFeatureString());
						newData.add(newDatum);
					//}
				}
			}
			if(printDebug) LogInfo.logs("\n------------------------------------------------");
		}
	
		return newData;
    }
    
    public List<BioDatum> setFeaturesTest(CoreMap sentence, Set<Tree> selectedEntities, String exampleID) {
    	// this is so that the feature factory code doesn't accidentally use the
    	// true label info
    	List<BioDatum> newData = new ArrayList<BioDatum>();

    	IdentityHashMap<Tree, EventType> eventNodes = Utils.getEventNodesFromSentence(sentence);
		//for(EventMention event: sentence.get(EventMentionsAnnotation.class)) {
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
				//for (String possibleLabel : labels) {
					if(node.isLeaf() || node.value().equals("ROOT") || !node.isPreTerminal() || 
							!(node.value().startsWith("JJR") || node.value().startsWith("JJS") ||node.value().startsWith("NN") || node.value().equals("JJ") || node.value().startsWith("VB")))
						continue;
					
					String type = eventNodes.keySet().contains(node) ? "E" : "O";
					
					BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), type, node, node, exampleID);
					newDatum.features = computeFeatures(sentence, node, selectedEntities);
					newData.add(newDatum);
					//prevLabel = newDatum.label;
				//}
		    //}
		}
    	return newData;
    }
    
    public FeatureVector computeFeatures(CoreMap sentence, Tree event, Set<Tree> entityNodes) {
		List<String> features = new ArrayList<String>();
		Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		IntPair eventSpan = event.getSpan();
				
		FeatureVector fv = basicFeatureFactory.computeFeatures(sentence, event);
				
		IdentityHashSet<IndexedWord> uniqueHeads = new IdentityHashSet<IndexedWord>();
		//Trying to look at entities to improve prediction
		if(addEntityFeatures) {
			String shortestPath = "";
			int pathLength = Integer.MAX_VALUE, numRelatedEntities = 0;
			int closestEntitySpanBefore = Integer.MAX_VALUE, closestEntitySpanAfter = Integer.MAX_VALUE;
			for(Tree entityNode:entityNodes) {
				if(entityNode==null)
					continue;
				IndexedWord entityIndexWord = Utils.findDependencyNode(sentence, entityNode);
				if(Utils.isNodesRelated(sentence, entityNode, event) && !uniqueHeads.contains(entityIndexWord)) {
					uniqueHeads.add(entityIndexWord);
					numRelatedEntities++;
				}
				
				List<String> nodesInPath = Trees.pathNodeToNode(event, entityNode, Trees.getLowestCommonAncestor(event, entityNode, root));
				//Okayish
				
				if(nodesInPath != null) {
					if(!(nodesInPath.contains("up-S") || nodesInPath.contains("up-SBAR")))
					{
						features.add("pathtoentities=" + StringUtils.join(nodesInPath, ","));
						
						if(pathLength > nodesInPath.size()) {
							shortestPath = StringUtils.join(nodesInPath, ",");
						}
					}
				}
				
				features.add("deppath=" + Utils.getDependencyPath(sentence, entityNode, event));
				
				
				IntPair entitySpan = entityNode.getSpan();
				if(entitySpan.getSource() < eventSpan.getSource()) {
					//features.add("entitybefore");
					if(Math.abs(entitySpan.getTarget() - eventSpan.getSource()) < closestEntitySpanBefore)
						closestEntitySpanBefore = Math.abs(entitySpan.getTarget() - eventSpan.getSource());
				}
				if(entitySpan.getSource() > eventSpan.getSource()) {
					//features.add("entityafter");
					if(Math.abs(entitySpan.getSource() - eventSpan.getTarget()) < closestEntitySpanBefore)
						closestEntitySpanBefore = Math.abs(entitySpan.getSource() - eventSpan.getTarget());
				}
			}
	
			if(closestEntitySpanAfter > Integer.MAX_VALUE)
				features.add("closestEntitySpanAfter=" + closestEntitySpanAfter);
			
			if(closestEntitySpanBefore > Integer.MAX_VALUE)
				features.add("closestEntitySpanBefore" + closestEntitySpanBefore);
			//Quite good
			features.add("numrelatedentities=" + (uniqueHeads.size() ));
			//not so great
			if(pathLength<Integer.MAX_VALUE) {
				features.add("splpath=" + shortestPath);
				features.add("splpathlength=" + pathLength);
			}
		}

		fv.add(features);
		return fv;
    }   
}
