package edu.stanford.nlp.bioprocess;

import java.util.*;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import fig.basic.LogInfo;

public class EventExtendedFeatureFactory extends FeatureExtractor {
	boolean printDebug = false, printAnnotations = false, printFeatures = false;
	Set<String> nominalizations = Utils.getNominalizedVerbs();
   
	public FeatureVector computeFeatures(CoreMap sentence, String tokenClass, Tree event) {
	    //LogInfo.logs("Current node's text - " + getText(event));
		
		List<String> features = new ArrayList<String>();
		String currentWord = event.value();
		List<Tree> leaves = event.getLeaves();
		Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		CoreLabel token = Utils.findCoreLabelFromTree(sentence, event);
		
		features.add("POS="+currentWord);
		features.add("lemma="+token.lemma());
		features.add("word=" + leaves.get(0));
		features.add("POSparentPOS="+currentWord + "," + event.parent(root).value());
		features.add("POSlemma=" + currentWord+","+token.lemma());
		features.add("path=" + Trees.pathFromRoot(event, root));
		
		//Trying to look at entities to improve prediction
		for(EntityMention m:sentence.get(EntityMentionsAnnotation.class)) {
			if(Utils.isNodesRelated(sentence, m.getTreeNode(), event));
				//features.add("Entityparent");
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

    public List<Datum> setFeaturesTrain(List<Example> data) {
		List<Datum> newData = new ArrayList<Datum>();
		
		for (Example ex : data) {
			if(printDebug || printAnnotations) LogInfo.logs("\n-------------------- " + ex.id + "---------------------");
			for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
				IdentityHashSet<Tree> eventNodes = Utils.getEventNodesFromSentence(sentence);
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
						if(node.isLeaf() || node.value().equals("ROOT") || !node.isPreTerminal())
							continue;
						
						String type = "O";
						
						//if ((entityNodes.contains(node) && Utils.getArgumentMentionRelation(event, node) != RelationType.NONE)) {// || Utils.isChildOfEntity(entityNodes, node)) {
							//type = "E";
						//}
						if (eventNodes.contains(node)){
							type = "E";
						}
						//if(printDebug) LogInfo.logs(type + " : " + node + ":" + node.getSpan());
						Datum newDatum = new Datum(Utils.getText(node), type, node, node);
						newDatum.features = computeFeatures(sentence, type, node);
						if(printFeatures) LogInfo.logs(Utils.getText(node) + ":" + newDatum.features);
						newData.add(newDatum);
					//}
				}
			}
			if(printDebug) LogInfo.logs("\n------------------------------------------------");
		}
	
		return newData;
    }
    
    public List<Datum> setFeaturesTest(CoreMap sentence) {
    	// this is so that the feature factory code doesn't accidentally use the
    	// true label info
    	List<Datum> newData = new ArrayList<Datum>();
    	List<String> labels = new ArrayList<String>();
    	Map<String, Integer> labelIndex = new HashMap<String, Integer>();

    	labelIndex.put("O", 0);
    	labelIndex.put("E", 1);
    	labels.add("O");
    	labels.add("E");


    	IdentityHashSet<Tree> eventNodes = Utils.getEventNodesFromSentence(sentence);
		//for(EventMention event: sentence.get(EventMentionsAnnotation.class)) {
			for(Tree node: sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
				for (String possibleLabel : labels) {
					if(node.isLeaf() || node.value().equals("ROOT") || !node.isPreTerminal())
						continue;
					
					String type = "O";
					
					if (eventNodes.contains(node)) {
						type = "E";
					}
					
					Datum newDatum = new Datum(Utils.getText(node), type, node, node);
					newDatum.features = computeFeatures(sentence, possibleLabel, node);
					newData.add(newDatum);
					//prevLabel = newDatum.label;
				}
		    //}
		}
    	return newData;
    }
}
