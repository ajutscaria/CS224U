package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.ArgumentRelation.EventType;
import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import fig.basic.LogInfo;

public class EventRelationFeatureFactory extends FeatureExtractor {

	private boolean printAnnotations = false, printDebug = false;

	public EventRelationFeatureFactory(boolean useLexicalFeatures) {
		super(useLexicalFeatures);
		// TODO Auto-generated constructor stub
	}
	
	private FeatureVector computeFeatures(CoreMap sentence, Tree treeNode,
			Tree treeNode2) {
		// TODO Auto-generated method stub
		return new FeatureVector();
	}

	@Override
	public List<BioDatum> setFeaturesTrain(List<Example> data) {
    	List<BioDatum> dataset = new ArrayList<BioDatum>();
		for (Example ex : data) {
			if(printDebug || printAnnotations) LogInfo.logs("\n-------------------- " + ex.id + "---------------------");
			for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
				if(printDebug) LogInfo.logs(sentence);
				if(printAnnotations) {
					LogInfo.logs("---Events--");
					for(EventMention event: sentence.get(EventMentionsAnnotation.class))
						LogInfo.logs(event.getValue());
				}
				List<EventMention> list = sentence.get(EventMentionsAnnotation.class);
				for(EventMention event1: list) {
					for(EventMention event2:list) {	
						String type = Utils.getArgumentMentionRelation(event1, event2.getTreeNode()).toString();
						BioDatum newDatum = new BioDatum(sentence, Utils.getText(event1.getTreeNode()) + "-" + Utils.getText(event2.getTreeNode()), type, event1, event2);
						newDatum.features = computeFeatures(sentence, event1.getTreeNode(), event2.getTreeNode());
						dataset.add(newDatum);
					//if(printFeatures) LogInfo.logs(Utils.getText(node) + ":" + newDatum.features.getFeatureString());
				}
			}
			if(printDebug) LogInfo.logs("\n------------------------------------------------");
			}
		}
	
		return dataset;
	}

	@Override
	public List<BioDatum> setFeaturesTest(CoreMap sentence, Set<Tree> selectedNodes) {
    	List<BioDatum> newData = new ArrayList<BioDatum>();
		for(Tree event1: selectedNodes) {
			for(Tree event2: selectedNodes) {
				String type = Utils.getArgumentMentionRelation(sentence, event1, event2).toString();
				BioDatum newDatum = new BioDatum(sentence, Utils.getText(event1) + "-" + Utils.getText(event2), type, event1, event2);
				newDatum.features = computeFeatures(sentence, event1, event2);
				newData.add(newDatum);
		    }
		}
    	return newData;
	}

}
