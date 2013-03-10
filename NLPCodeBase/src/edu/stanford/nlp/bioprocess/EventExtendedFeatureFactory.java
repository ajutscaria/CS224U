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
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.trees.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.IntPair;
import fig.basic.LogInfo;
import edu.stanford.nlp.util.StringUtils;

public class EventExtendedFeatureFactory extends FeatureExtractor {
	boolean printDebug = false, printAnnotations = false, printFeatures = false;
	boolean addEntityFeatures = true;
	Set<String> nominalizations = Utils.getNominalizedVerbs();
   
	public FeatureVector computeFeatures(CoreMap sentence, String tokenClass, Tree event) {
	    //LogInfo.logs("Current node's text - " + getText(event));
		
		List<String> features = new ArrayList<String>();
		String currentWord = event.value();
		List<Tree> leaves = event.getLeaves();
		Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
		CoreLabel token = Utils.findCoreLabelFromTree(sentence, event);
		List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
		IndexedWord word = Utils.findDependencyNode(sentence, event);
		Tree parent = event.parent(root);
		IntPair eventSpan = event.getSpan();
		
		String parentCFGRule = parent.value() + "->";
		for(Tree n:parent.getChildrenAsList()) {
			parentCFGRule += n.value() + "|";
		}
		parentCFGRule = parentCFGRule.trim();
		
		features.add("POS="+currentWord);
		features.add("lemma="+token.lemma());
		features.add("word="+token.originalText());
		features.add("POSword=" + currentWord+","+leaves.get(0));
		features.add("POSparentPOS="+currentWord + "," + event.parent(root).value());
		features.add("POSlemma=" + currentWord+","+token.lemma());
		features.add("path=" + StringUtils.join(Trees.pathNodeToNode(root, event, root), ",").replace("up-ROOT,down-ROOT,", ""));
		features.add("POSparentrule=" + currentWord+","+parentCFGRule);
		
		for(SemanticGraphEdge e: graph.getIncomingEdgesSorted(word)) {
			features.add("depEdgeIn="+ e.getRelation() + "," + e.getSource().toString().split("-")[1]);
		}
		
		//for(SemanticGraphEdge e: graph.getOutEdgesSorted(word)) {
		//	features.add("depedgeout="+ e.getRelation() + "," + e.getSource().toString().split("-")[1]);
			//System.out.println(e.getRelation());
			//System.out.println(e.getSource().toString());
		//}
		
		//Trying to look at entities to improve prediction
		if(addEntityFeatures) {
			String shortestPath = "";
			int pathLength = Integer.MAX_VALUE, numRelatedEntities = 0;
			int closestEntitySpanBefore = Integer.MAX_VALUE, closestEntitySpanAfter = Integer.MAX_VALUE;
			for(EntityMention m:sentence.get(EntityMentionsAnnotation.class)) {
				if(m.getTreeNode()==null)
					continue;
				if(Utils.isNodesRelated(sentence, m.getTreeNode(), event)) 
					numRelatedEntities++;
				
				List<String> nodesInPath = Trees.pathNodeToNode(event, m.getTreeNode(), Trees.getLowestCommonAncestor(event, m.getTreeNode(), root));
				//Okayish
				if(!(nodesInPath.contains("up-S") || nodesInPath.contains("up-SBAR")))
					features.add("pathtoentities=" + StringUtils.join(nodesInPath, ","));	
				
				if(pathLength > nodesInPath.size()) {
					shortestPath = StringUtils.join(nodesInPath, ",");
				}
				IntPair entitySpan = m.getTreeNode().getSpan();
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
	
			//if(closestEntitySpanAfter > Integer.MAX_VALUE)
			//	features.add("closestEntitySpanAfter=" + closestEntitySpanAfter);
			
			//if(closestEntitySpanBefore > Integer.MAX_VALUE)
			//	features.add("closestEntitySpanBefore" + closestEntitySpanBefore);
			//Quite good
			features.add("numrelatedentities=" + (numRelatedEntities ));
			//not so great
			if(pathLength<Integer.MAX_VALUE) {
				//features.add("splpath=" + shortestPath);
				//features.add("splpathlength=" + pathLength);
			}
		}
		
		//Nominalization did not give much improvement
		/*if(nominalizations.contains(leaves.get(0).value())) {
			//LogInfo.logs("Adding nominalization - " + leaves.get(0));
			//features.add("nominalization");
		}*/
		//features.add("endsining=" + token.lemma() + "," + leaves.get(0).value().endsWith("ing"));
		//Cannot use this feature when looking at all tree nodes as candidates
		//features.add("POSparentPOSgrandparent="+currentWord + "," + event.parent(root).value() + "," + event.parent(root).parent(root).value());
		//Doesn't seem to work as expected even though the event triggers are mostly close to root in dependency tree.
		//features.add("POSdepdepth=" + currentWord + "," + Utils.findDepthInDependencyTree(sentence, event));
		
		String classString = "class=" + tokenClass + ",";
		List<String> updatedFeatures = new ArrayList<String>();
		for(String feature:features)
			updatedFeatures.add(classString + feature);
	
		//return null;
		FeatureVector fv = new FeatureVector(updatedFeatures);
		return fv;
    }

    public List<Datum> setFeaturesTrain(List<Example> data) {
		List<Datum> newData = new ArrayList<Datum>();
		
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
								!(node.value().equals("NN") || node.value().equals("NNS") || node.value().equals("JJ") || node.value().startsWith("VB")))
							continue;
						
						String type = "O";
						
						//if ((entityNodes.contains(node) && Utils.getArgumentMentionRelation(event, node) != RelationType.NONE)) {// || Utils.isChildOfEntity(entityNodes, node)) {
							//type = "E";
						//}
						if (eventNodes.keySet().contains(node)){
							type = "E";
						}
						//if(printDebug) LogInfo.logs(type + " : " + node + ":" + node.getSpan());
						Datum newDatum = new Datum(sentence, Utils.getText(node), type, node, node);
						newDatum.features = computeFeatures(sentence, type, node);
						if(printFeatures) LogInfo.logs(Utils.getText(node) + ":" + newDatum.features.getFeatureString());
						newData.add(newDatum);
					//}
				}
			}
			if(printDebug) LogInfo.logs("\n------------------------------------------------");
		}
	
		return newData;
    }
    
    public List<Datum> setFeaturesTest(CoreMap sentence, Set<Tree> selectedEntities) {
    	// this is so that the feature factory code doesn't accidentally use the
    	// true label info
    	List<Datum> newData = new ArrayList<Datum>();
    	List<String> labels = new ArrayList<String>();
    	Map<String, Integer> labelIndex = new HashMap<String, Integer>();

    	labelIndex.put("O", 0);
    	labelIndex.put("E", 1);
    	labels.add("O");
    	labels.add("E");


    	IdentityHashMap<Tree, EventType> eventNodes = Utils.getEventNodesFromSentence(sentence);
		//for(EventMention event: sentence.get(EventMentionsAnnotation.class)) {
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
				for (String possibleLabel : labels) {
					if(node.isLeaf() || node.value().equals("ROOT") || !node.isPreTerminal() || 
							!(node.value().equals("NN") || node.value().equals("NNS") || node.value().equals("JJ") || node.value().startsWith("VB")))
						continue;
					
					String type = "O";
					
					if (eventNodes.keySet().contains(node)) {
						type = "E";
					}
					
					Datum newDatum = new Datum(sentence, Utils.getText(node), type, node, node);
					newDatum.features = computeFeatures(sentence, possibleLabel, node, selectedEntities);
					newData.add(newDatum);
					//prevLabel = newDatum.label;
				}
		    //}
		}
    	return newData;
    }
    
    
    public FeatureVector computeFeatures(CoreMap sentence, String tokenClass, Tree event, Set<Tree> entityNodes) {
	    //System.out.println("Extracting features!!!");
		
		List<String> features = new ArrayList<String>();
		String currentWord = event.value();
		List<Tree> leaves = event.getLeaves();
		Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
		CoreLabel token = Utils.findCoreLabelFromTree(sentence, event);
		List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
		IndexedWord word = Utils.findDependencyNode(sentence, event);
		Tree parent = event.parent(root);
		IntPair eventSpan = event.getSpan();
		
		String parentCFGRule = parent.value() + "->";
		for(Tree n:parent.getChildrenAsList()) {
			parentCFGRule += n.value() + "|";
		}
		parentCFGRule = parentCFGRule.trim();
		
		features.add("POS="+currentWord);
		features.add("lemma="+token.lemma());
		features.add("word="+token.originalText());
		features.add("POSword=" + currentWord+","+leaves.get(0));
		features.add("POSparentPOS="+currentWord + "," + event.parent(root).value());
		features.add("POSlemma=" + currentWord+","+token.lemma());
		features.add("path=" + StringUtils.join(Trees.pathNodeToNode(root, event, root), ",").replace("up-ROOT,down-ROOT,", ""));
		features.add("POSparentrule=" + currentWord+","+parentCFGRule);
		
		for(SemanticGraphEdge e: graph.getIncomingEdgesSorted(word)) {
			features.add("depedgein="+ e.getRelation() + "," + e.getSource().toString().split("-")[1]);
		}
		
		//for(SemanticGraphEdge e: graph.getOutEdgesSorted(word)) {
		//	features.add("depedgeout="+ e.getRelation() + "," + e.getSource().toString().split("-")[1]);
			//System.out.println(e.getRelation());
			//System.out.println(e.getSource().toString());
		//}
		
		//Trying to look at entities to improve prediction
		if(addEntityFeatures) {
			String shortestPath = "";
			int pathLength = Integer.MAX_VALUE, numRelatedEntities = 0;
			int closestEntitySpanBefore = Integer.MAX_VALUE, closestEntitySpanAfter = Integer.MAX_VALUE;
			for(Tree entityNode:entityNodes) {
				if(entityNode==null)
					continue;
				if(Utils.isNodesRelated(sentence, entityNode, event)) 
					numRelatedEntities++;
				
				List<String> nodesInPath = Trees.pathNodeToNode(event, entityNode, Trees.getLowestCommonAncestor(event, entityNode, root));
				//Okayish
				if(!(nodesInPath.contains("up-S") || nodesInPath.contains("up-SBAR")))
					features.add("pathtoentities=" + StringUtils.join(nodesInPath, ","));	
				
				if(pathLength > nodesInPath.size()) {
					shortestPath = StringUtils.join(nodesInPath, ",");
				}
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
	
			//if(closestEntitySpanAfter > Integer.MAX_VALUE)
			//	features.add("closestEntitySpanAfter=" + closestEntitySpanAfter);
			
			//if(closestEntitySpanBefore > Integer.MAX_VALUE)
			//	features.add("closestEntitySpanBefore" + closestEntitySpanBefore);
			//Quite good
			features.add("numrelatedentities=" + (numRelatedEntities ));
			//not so great
			if(pathLength<Integer.MAX_VALUE) {
				//features.add("splpath=" + shortestPath);
				//features.add("splpathlength=" + pathLength);
			}
		}
		
		//Nominalization did not give much improvement
		/*if(nominalizations.contains(leaves.get(0).value())) {
			//LogInfo.logs("Adding nominalization - " + leaves.get(0));
			//features.add("nominalization");
		}*/
		//features.add("endsining=" + token.lemma() + "," + leaves.get(0).value().endsWith("ing"));
		//Cannot use this feature when looking at all tree nodes as candidates
		//features.add("POSparentPOSgrandparent="+currentWord + "," + event.parent(root).value() + "," + event.parent(root).parent(root).value());
		//Doesn't seem to work as expected even though the event triggers are mostly close to root in dependency tree.
		//features.add("POSdepdepth=" + currentWord + "," + Utils.findDepthInDependencyTree(sentence, event));
		
		String classString = "class=" + tokenClass + ",";
		List<String> updatedFeatures = new ArrayList<String>();
		for(String feature:features)
			updatedFeatures.add(classString + feature);
	
		FeatureVector fv = new FeatureVector(updatedFeatures);
		return fv;
    }

}
