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
	HashMap<String, String> verbForms = Utils.getVerbForms();
   
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
						BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), type, node, node);
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
    
    public List<BioDatum> setFeaturesTest(CoreMap sentence, Set<Tree> selectedEntities) {
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
					
					BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), type, node, node);
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
		String currentWord = event.value();
		//List<Tree> leaves = event.getLeaves();
		Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
		CoreLabel token = Utils.findCoreLabelFromTree(sentence, event);
		List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
		int currentTokenIndex = event.getSpan().getSource();
		
		//if (currentTokenIndex < tokens.size()-1){
			//System.out.println(String.format("current word is %s, next word is %s",token.originalText(), tokens.get(event.getSpan().getSource()+1).originalText()));
		//}
		IndexedWord word = Utils.findDependencyNode(sentence, event);
		Tree parent = event.parent(root);
		IntPair eventSpan = event.getSpan();
		
		String parentCFGRule = parent.value() + "->";
		for(Tree n:parent.getChildrenAsList()) {
			parentCFGRule += n.value() + "|";
		}
		parentCFGRule = parentCFGRule.trim();
		
		//features.add("POS="+currentWord);
		/*if (Utils.findDepthInDependencyTree(sentence, event)==0)
			features.add("root=true,POS="+currentWord);
		*/
		String text = token.originalText();
		if(verbForms.containsKey(text)) {
			features.add("lemma="+verbForms.get(text));
			features.add("word="+verbForms.get(text));
			//features.add("word="+token.originalText());
		}
		else {
			features.add("lemma="+token.lemma().toLowerCase());
			features.add("word="+token.originalText());
		}
		
		//features.add("POSword=" + currentWord+","+leaves.get(0));
		//features.add("POSparentPOS="+ currentWord + "," + event.parent(root).value());
		features.add("POSlemma=" + currentWord+","+token.lemma());
		//if(currentWord.startsWith("VB"))
		//	features.add("verb");
		features.add("ParentPOS=" + parent.value());
		features.add("path=" + StringUtils.join(Trees.pathNodeToNode(root, event, root), ",").replace("up-ROOT,down-ROOT,", ""));
		features.add("POSparentrule=" + currentWord+","+parentCFGRule);
		
		for(SemanticGraphEdge e: graph.getIncomingEdgesSorted(word)) {
			//features.add("depedgein="+ e.getRelation());// + "," + e.getSource().toString().split("-")[1]);
			//features.add("depedgeinword="+currentWord +"," + e.getRelation() + "," + e.getSource().toString().split("-")[0] + ","+ e.getSource().toString().split("-")[1]);
			//LogInfo.logs("depedgeinword="+currentWord +"," + e.getRelation() + "," + e.getSource().toString().split("-")[0] + ","+ e.getSource().toString().split("-")[1]);
		}
		
		for(SemanticGraphEdge e: graph.getOutEdgesSorted(word)) {
			if(e.getRelation().toString().equals("advmod") && (currentWord.startsWith("VB") || nominalizations.contains(text)))
				features.add("advmod:" + e.getTarget());
				//LogInfo.logs("TIMEE : " + e.getRelation() + ":" + e.getTarget());
			//features.add("depedgein="+ e.getRelation() + "," + e.getTarget().toString().split("-")[1]);//need to deal with mult children same tag?
			//features.add("depedgeinword="+currentWord +"," + e.getRelation() + "," + e.getSource().toString().split("-")[0] + ","+ e.getSource().toString().split("-")[1]);
			//LogInfo.logs("depedgeinword="+currentWord +"," + e.getRelation() + "," + e.getSource().toString().split("-")[0] + ","+ e.getSource().toString().split("-")[1]);
		}

		if(nominalizations.contains(token.value())) {
			//LogInfo.logs("Adding nominalization - " + leaves.get(0));
			features.add("nominalization");
		}
		
		String consecutiveTypes = "";
		if(currentTokenIndex > 0)
			consecutiveTypes += tokens.get(currentTokenIndex-1).get(PartOfSpeechAnnotation.class);
		consecutiveTypes += currentWord;
		if(currentTokenIndex < tokens.size() - 1)
			consecutiveTypes += tokens.get(currentTokenIndex+1).get(PartOfSpeechAnnotation.class);
		features.add("consecutivetypes="+consecutiveTypes);
		
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
				if(!(nodesInPath.contains("up-S") || nodesInPath.contains("up-SBAR")))
					features.add("pathtoentities=" + StringUtils.join(nodesInPath, ","));	
				
				features.add("deppath=" + Utils.getDependencyPath(sentence, entityNode, event));
				
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
		features.add("bias");
		FeatureVector fv = new FeatureVector(features);
		return fv;
    }   
}
