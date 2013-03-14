package edu.stanford.nlp.bioprocess;


import java.util.*;

import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;
import fig.basic.LogInfo;

public class SRLFeatureFactory extends FeatureExtractor {
	boolean printDebug = false, printAnnotations = false, printFeatures = true;
	HashMap<String, Integer> relCount = new HashMap<String, Integer>();
	Index labelIndex;
	
	public SRLFeatureFactory() {
	}
	
	public SRLFeatureFactory(Index labelIndex) {
		this.labelIndex = labelIndex;
	}

    public FeatureVector computeFeatures(CoreMap sentence, String tokenClass, Tree entity,  Tree event) {
	    //Tree event = eventMention.getTreeNode();
		List<String> features = new ArrayList<String>();
	
		Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		boolean dependencyExists = Utils.isNodesRelated(sentence, entity, event);
		
		CoreLabel token = Utils.findCoreLabelFromTree(sentence, entity);
		List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
		
		//features.add("PrevWordPOS="+tokens.get(token.index()-1).get(PartOfSpeechAnnotation.class));
		features.add("EntityEvent="+Utils.getText(entity)+","+Utils.getText(event));
		//System.out.println("Adding "+entity.getLeaves()+"-"+event.getLeaves());

//		features.add("EntPOSDepRel=" + entity.value() + ","  + dependencyExists);
//		features.add("EntHeadEvtPOS="+Utils.findCoreLabelFromTree(sentence, entity).lemma() + "," + event.preTerminalYield().get(0).value());
//		features.add("PathEntToEvt=" + Trees.pathNodeToNode(event, entity, root));
//		features.add("EntHeadEvtHead=" + entity.headTerminal(new CollinsHeadFinder()) + "," + event.getLeaves().get(0));
//		features.add("EntNPAndRelatedToEvt=" + (entity.value().equals("NP") && Utils.isNodesRelated(sentence, entity, event)));
		
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
				//IdentityHashSet<Tree> entityNodes = Utils.getEntityNodesFromSentence(sentence);
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
						
						String type = Utils.getArgumentMentionRelation(event, node).toString();
						
						if (relCount.containsKey(type)) {
							relCount.put(type, relCount.get(type)+1);
						} else {
							relCount.put(type, 1);
						}
						
//						if ((entityNodes.contains(node) && Utils.getArgumentMentionRelation(event, node) != RelationType.NONE)) {// || Utils.isChildOfEntity(entityNodes, node)) {
//							type = "E";
//						}
						
						if(printDebug) LogInfo.logs(type + " : " + node + ":" + node.getSpan());
	//					if((entityNodes.contains(node))){// || (Utils.isChildOfEntity(entityNodes, node) && node.value().startsWith("NN"))) {
	//						type = "E";
	//					}
						
						Datum newDatum = new Datum(sentence, Utils.getText(node), type, node, event.getTreeNode());
						newDatum.features = computeFeatures(sentence, type, node, event.getTreeNode());
						//System.out.println(Utils.getText(node) + ":" + newDatum.features.features.toString());
						if(printFeatures) {
							LogInfo.logs(Utils.getText(node) + ":" + newDatum.features.getFeatureString());
						}
						newData.add(newDatum);
				}
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
    	
		for(Tree eventNode: predictedEvents) {
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
				for (int i=0; i<this.labelIndex.size(); i++) {
					String possibleLabel = (String) this.labelIndex.get(i);
					if(node.isLeaf() || node.value().equals("ROOT"))
						continue;
					
					String type = Utils.getArgumentMentionRelation(sentence, eventNode, node).toString();
					
					Datum newDatum = new Datum(sentence, Utils.getText(node), type, node, eventNode);
					newDatum.features = computeFeatures(sentence, possibleLabel, node, eventNode);
					newData.add(newDatum);
				}
		    }
		}
    	return newData;
    }
    
    public String getMostCommonRelationType() {
    	int maximum = -1;
    	String popularRelation = null;
    	for (String relType : relCount.keySet()) {
    		if (relType.equals(RelationType.NONE.toString())) {
    			continue;
    		}
    		if (relCount.get(relType) > maximum) {
    			maximum = relCount.get(relType);
    			popularRelation = relType;
    		}
    	}
    	return popularRelation;
    }
}
