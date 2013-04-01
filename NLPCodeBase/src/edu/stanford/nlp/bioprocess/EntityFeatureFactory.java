package edu.stanford.nlp.bioprocess;


import java.util.*;

import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import fig.basic.LogInfo;

public class EntityFeatureFactory extends FeatureExtractor {
	public EntityFeatureFactory(boolean useLexicalFeatures) {
		super(useLexicalFeatures);
	}


	boolean printDebug = false, printAnnotations = false, printFeatures = false;

    public FeatureVector computeFeatures(CoreMap sentence, Tree entity,  Tree event) {
	    //Tree event = eventMention.getTreeNode();
		List<String> features = new ArrayList<String>();
		//List<Tree> leaves = entity.getLeaves();
		Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		boolean dependencyExists = Utils.isNodesRelated(sentence, entity, event);
		String depPath = Utils.getDependencyPath(sentence, entity, event);
		Tree parent = entity.parent(root);
		String parentCFGRule = parent.value() + "->";
		for(Tree n:parent.getChildrenAsList()) {
			parentCFGRule += n.value() + "|";
		}
		parentCFGRule = parentCFGRule.trim();
		
		//features.add("dep="+dependencyExists);
		//features.add("EntCat="+entity.value());
		//features.add("EntHead=" + entity.headTerminal(new CollinsHeadFinder()));
		features.add("EvtLemma="+event.getLeaves().get(0).value());
		features.add("EntCatDepRel=" + entity.value() + ","  + dependencyExists);
		features.add("EntHeadEvtPOS="+Utils.findCoreLabelFromTree(sentence, entity).lemma() + "," + event.preTerminalYield().get(0).value());
		features.add("EvtToEntDepPath=" + ((depPath.equals("")||depPath.equals("[]")) ? 0 :depPath.split(",").length));
		features.add("EntHeadEvtHead=" + entity.headTerminal(new CollinsHeadFinder()) + "," + event.getLeaves().get(0));		
		//features.add("PathEntToEvt=" + Trees.pathNodeToNode(event, entity, root));
		features.add("EntNPAndRelatedToEvt=" + (entity.value().equals("NP") && Utils.isNodesRelated(sentence, entity, event)));
		//features.add("parentrule=" + parentCFGRule);
		//USE LEMMA everywhere.
		//Try before/after
		
		//features.add("EntPOSEntHeadEvtPOS=" + entity.value() + "," + entity.headTerminal(new CollinsHeadFinder()) + "," + event.preTerminalYield().get(0).value());
		//features.add("EntPOSEvtPOSDepRel=" + entity.value() + "," +event.preTerminalYield().get(0).value() + ","  + dependencyExists);
		//features.add("EntPOSEntParentPOSEvtPOS=" + entity.value() + "," + entity.headTerminal(new CollinsHeadFinder()) + "," + event.preTerminalYield().get(0).value());
		//features.add("EntLastWordEvtPOS="+leaves.get(leaves.size()-1)+","+event.preTerminalYield().get(0).value());
		//features.add("PathEntToAncestor="+Trees.pathNodeToNode(entity, Trees.getLowestCommonAncestor(entity, event, root), root));
		//features.add("PathEntToRoot="+Trees.pathNodeToNode(entity, root, root));
		//features.add("EntParentPOSEvtPOS=" + entity.headTerminal(new CollinsHeadFinder()) + "," + event.preTerminalYield().get(0).value());
		
		
		
		//This feature did not work surprisingly. Maybe because the path from ancestor to event might lead to a lot of different variations.
		//features.add("PathAncestorToEvt="+Trees.pathNodeToNode(Trees.getLowestCommonAncestor(entity, event, root), event, root));
		//This is a bad feature too.
		//features.add("EvtPOSDepRel=" + event.preTerminalYield().get(0).value() + ","  + dependencyExists);
		//Not a good feature too.
		//features.add("EntPOSEvtPOS=" + entity.value() + "," + event.preTerminalYield().get(0).value());
		features.add("bias");
		FeatureVector fv = new FeatureVector(features);
		return fv;
    }

    public List<BioDatum> setFeaturesTrain(List<Example> data) {
    	List<BioDatum> newData = new ArrayList<BioDatum>();
	
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
				//SemanticGraph dependencies = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
				//LogInfo.logs(dependencies);
				for(EventMention event: sentence.get(EventMentionsAnnotation.class)) {
					if(printDebug) LogInfo.logs("-------Event - " + event.getTreeNode()+ "--------");
					for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
						if(node.isLeaf()||node.value().equals("ROOT"))
							continue;
						
						String type = "O";
						
						if ((entityNodes.contains(node) && Utils.getArgumentMentionRelation(event, node) != RelationType.NONE)) {// || Utils.isChildOfEntity(entityNodes, node)) {
							type = "E";
						}
						if(printDebug) LogInfo.logs(type + " : " + node + ":" + node.getSpan());
	//					if((entityNodes.contains(node))){// || (Utils.isChildOfEntity(entityNodes, node) && node.value().startsWith("NN"))) {
	//						type = "E";
	//					}
						
						BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), type, node, event.getTreeNode(), Utils.getArgumentMentionRelation(event, node).toString());
						newDatum.features = computeFeatures(sentence, node, event.getTreeNode());
						if(printFeatures) LogInfo.logs(Utils.getText(node) + ":" + newDatum.features);
						newData.add(newDatum);
				}
			}
		}
		if(printDebug) LogInfo.logs("\n------------------------------------------------");
	}

	return newData;
    }
    
    
    public List<BioDatum> setFeaturesTest(CoreMap sentence, Set<Tree> predictedEvents) {
    	// this is so that the feature factory code doesn't accidentally use the
    	// true label info
    	List<BioDatum> newData = new ArrayList<BioDatum>();

    	IdentityHashSet<Tree> entityNodes = Utils.getEntityNodesFromSentence(sentence);
		for(Tree eventNode: predictedEvents) {
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
				if(node.isLeaf() || node.value().equals("ROOT"))
					continue;
				String type = (entityNodes.contains(node) && Utils.getArgumentMentionRelation(sentence, eventNode, node) != RelationType.NONE) ? "E" : "O";
				
				BioDatum newDatum = new BioDatum(sentence, Utils.getText(node), type, node, eventNode, Utils.getArgumentMentionRelation(sentence, eventNode, node).toString());
				newDatum.features = computeFeatures(sentence, node, eventNode);
				newData.add(newDatum);
		    }
		}
    	return newData;
    }
}
