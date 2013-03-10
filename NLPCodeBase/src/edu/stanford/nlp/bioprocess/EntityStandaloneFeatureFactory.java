package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.StringUtils;
import fig.basic.LogInfo;

public class EntityStandaloneFeatureFactory extends FeatureExtractor {

	boolean printDebug = true, printAnnotations = false, printFeatures = false;

    public FeatureVector computeFeatures(CoreMap sentence, String tokenClass, Tree entity,  Tree event) {
	    //Tree event = eventMention.getTreeNode();
    	Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		List<String> features = new ArrayList<String>();
		IndexedWord word = Utils.findDependencyNode(sentence, entity);
		Tree parent = entity.parent(root);
		String currentWord = entity.value();
		CoreLabel token = Utils.findCoreLabelFromTree(sentence, entity);
		List<Tree> leaves = entity.getLeaves();
		SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
		String parentCFGRule = parent.value() + "->";
		for(Tree n:parent.getChildrenAsList()) {
			parentCFGRule += n.value() + " ";
		}
		parentCFGRule = parentCFGRule.trim();
		
		//features.add("POS="+currentWord);
		features.add("lemma="+token.lemma());
		features.add("word="+token.originalText().toLowerCase());
		//features.add("firstword=" + leaves.get(0));
		//features.add("lastword=" + leaves.get(leaves.size()-1));
		//features.add("POSparentPOS="+currentWord + "," + entity.parent(root).value());
		//features.add("POSlemma=" + currentWord+","+token.lemma());
		features.add("path=" + StringUtils.join(Trees.pathNodeToNode(root, entity, root), ",").replace("up-ROOT,down-ROOT,", ""));
		features.add("POSparentrule=" + currentWord+","+parentCFGRule);
		
		//for(SemanticGraphEdge e: graph.getIncomingEdgesSorted(word)) {
		//	features.add("depedgein="+ e.getRelation() + "," + e.getSource().toString().split("-")[1]);
		//}
		
		
		//This feature did not work surprisingly. Maybe because the path from ancestor to event might lead to a lot of different variations.
		//features.add("PathAncestorToEvt="+Trees.pathNodeToNode(Trees.getLowestCommonAncestor(entity, event, root), event, root));
		//This is a bad feature too.
		//features.add("EvtPOSDepRel=" + event.preTerminalYield().get(0).value() + ","  + dependencyExists);
		//Not a good feature too.
		//features.add("EntPOSEvtPOS=" + entity.value() + "," + event.preTerminalYield().get(0).value());
		String classString = "class=" + tokenClass + ",";
		List<String> updatedFeatures = new ArrayList<String>();
		for(String feature:features)
			updatedFeatures.add(classString + feature);
	
		FeatureVector fv = new FeatureVector(updatedFeatures);
		return fv;
    }

    public List<Datum> setFeaturesTrain(List<Example> data) {
    	List<Datum> newData = new ArrayList<Datum>();
	
		for (Example ex : data) {
			if(printDebug || printAnnotations) LogInfo.logs("\n-------------------- " + ex.id + "---------------------");
			for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
				IdentityHashSet<Tree> entityNodes = Utils.getEntityNodesFromSentence(sentence);
				if(printDebug){
					LogInfo.logs(sentence);
					sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
				}
				if(printAnnotations) {
					LogInfo.logs("---Events--");
					for(EventMention event: sentence.get(EventMentionsAnnotation.class))
						LogInfo.logs(event.getValue());
					LogInfo.logs("---Entities--");
					for(EntityMention entity: sentence.get(EntityMentionsAnnotation.class)) {
						if(entity.getTreeNode() != null)
							LogInfo.logs(entity.getTreeNode() + ":" + entity.getTreeNode().getSpan());
						else
							LogInfo.logs("Couldn't find node:" + entity.getValue());
					}
				}
				
				for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
					if(node.isLeaf()||node.value().equals("ROOT"))
						continue;
					
					String type = "O";
					
					if (entityNodes.contains(node)) {// || Utils.isChildOfEntity(entityNodes, node)) {
						type = "E";
					}
					if(printDebug) LogInfo.logs(type + " : " + node + ":" + node.getSpan());
//					if((entityNodes.contains(node))){// || (Utils.isChildOfEntity(entityNodes, node) && node.value().startsWith("NN"))) {
//						type = "E";
//					}
					
					Datum newDatum = new Datum(sentence, Utils.getText(node), type, node, null);
					newDatum.features = computeFeatures(sentence, type, node, null);
					if(printFeatures) LogInfo.logs(Utils.getText(node) + ":" + newDatum.features.getFeatureString());
					newData.add(newDatum);
				}
			
		}
		if(printDebug) LogInfo.logs("\n------------------------------------------------");
	}

	return newData;
    }
    
    
    public List<Datum> setFeaturesTest(CoreMap sentence, Set<Tree> predictedEvents) {
    	// this is so that the feature factory code doesn't accidentally use the
    	// true label info
    	List<Datum> newData = new ArrayList<Datum>();
    	List<String> labels = new ArrayList<String>();
    	Map<String, Integer> labelIndex = new HashMap<String, Integer>();

    	labelIndex.put("O", 0);
    	labelIndex.put("E", 1);
    	labels.add("O");
    	labels.add("E");


    	IdentityHashSet<Tree> entityNodes = Utils.getEntityNodesFromSentence(sentence);
		for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
			for (String possibleLabel : labels) {
				if(node.isLeaf() || node.value().equals("ROOT"))
					continue;
				
				String type = "O";
				
				if (entityNodes.contains(node)) {
					type = "E";
				}
				
				Datum newDatum = new Datum(sentence, Utils.getText(node), type, node, null);
				newDatum.features = computeFeatures(sentence, possibleLabel, node, null);
				newData.add(newDatum);
				//prevLabel = newDatum.label;
			}
	    }

    	return newData;
    }
}