package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.ArgumentRelation.EventType;
import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;

public class EventRelationFeatureFactory {

	private boolean printAnnotations = false, printDebug = false;
	private boolean useLexicalFeatures;
	HashMap<String, String> verbForms = Utils.getVerbForms();

	public EventRelationFeatureFactory(boolean useLexicalFeatures) {
		this.useLexicalFeatures = useLexicalFeatures;
		// TODO Auto-generated constructor stub
	}
	
	private FeatureVector computeFeatures(Example example, EventMention event1, EventMention event2) {
		List<String> features = new ArrayList<String>();
		
		//Is event2 immediately after event1?
		features.add("isImmediatelyAfter:" + Utils.isEventNextInOrder(example.gold.get(EventMentionsAnnotation.class), event1, event2));
		
		//Add words in between two event mentions.
		List<String> wordsInBetween = Utils.findWordsInBetween(example, event1, event2);
		for(String word:wordsInBetween)
			features.add("wordsInBetween:" + word);
		
		//POS tags of both events
		features.add("POS:" + event1.getTreeNode().value() + "+" + event2.getTreeNode().value());
		
		//Lemmas of both events
		String lemma1 = Utils.findCoreLabelFromTree(event1.getSentence(), event1.getTreeNode()).lemma().toLowerCase();
		if(verbForms.containsKey(lemma1)) {
			lemma1 = verbForms.get(lemma1);
		}
		String lemma2 = Utils.findCoreLabelFromTree(event2.getSentence(), event2.getTreeNode()).lemma().toLowerCase();
		if(verbForms.containsKey(lemma2)) {
			lemma2 = verbForms.get(lemma2);
		}
		features.add("Lemma:" + lemma1 + "+" + lemma2);
		features.add("eventLemmasSame:" + lemma1.equals(lemma2));
		
		//Number of sentences and words between two event mentions.
		Pair<Integer, Integer> counts =  Utils.findNumberOfSentencesAndWordsBetween(example, event1, event2);
		features.add("numSentencesInBetween:" +counts.first());
		features.add("numWordsInBetween:" +counts.first());
		
		//Word before first event and word after second event
		String wordBefore = Utils.findWordBefore(event1), wordAfter = Utils.findWordAfter(event2);
		if(wordBefore != null)
			features.add("wordBeforeFirstEvent:" + wordBefore);
		if(wordAfter != null)
			features.add("wordAfterFirstEvent:" + wordAfter);
		
		features.add("bias");
		
		FeatureVector fv = new FeatureVector(features);
		return fv;
	}

	public List<BioDatum> setFeaturesTrain(List<Example> data) {
    	List<BioDatum> dataset = new ArrayList<BioDatum>();
		for (Example ex : data) {
			if(printDebug || printAnnotations) LogInfo.logs("\n-------------------- " + ex.id + "---------------------");
			if(printDebug) LogInfo.logs(ex.id);
			if(printAnnotations) {
				LogInfo.logs("---Events-Event--");
				for(EventMention evt:ex.gold.get(EventMentionsAnnotation.class)) {
					for(ArgumentRelation rel:evt.getArguments()) {
						  if(rel.mention instanceof EventMention) { 
							  LogInfo.logs(evt.getTreeNode() + "-" + rel.mention.getTreeNode() + "-->" + rel.type);
						  }
					}
				}
			}
			List<EventMention> list = ex.gold.get(EventMentionsAnnotation.class);
			List<EventMention> alreadyConsidered = new ArrayList<EventMention>();
			for(EventMention event1: list) {
				alreadyConsidered.add(event1);
				for(EventMention event2:list) {	
					if(!alreadyConsidered.contains(event2)) {
						String type = Utils.getEventEventRelation(ex.gold, event1.getTreeNode(), event2.getTreeNode()).toString();
						BioDatum newDatum = new BioDatum(null, Utils.getText(event1.getTreeNode()) + "-" + Utils.getText(event2.getTreeNode()), type, event1, event2);
						newDatum.features = computeFeatures(ex, event1, event2);
						dataset.add(newDatum);
					}
				}
			}
			if(printDebug) LogInfo.logs("\n------------------------------------------------");
		}
	
		return dataset;
	}

	/*
	 * We order events in the same order as in they appear in the paragraph (This is done while the data is read in).
	 * Now, for each unique events ei, ej, where i<j always, we store the relation in gold labels
	 * For the relation ei R ej, we read it as ei is R of ej. So, 'causes' and 'enables' remains same if first event appears first in paragraph.
	 *  But, we need to swap 'next-event', 'super-event' with 'previous-event' and 'sub-event' respectively if the first event appears first in the paragraph.
	 */
	public List<BioDatum> setFeaturesTest(Example example, List<EventMention> list) {
    	List<BioDatum> newData = new ArrayList<BioDatum>();
    	List<EventMention> alreadyConsidered = new ArrayList<EventMention>();
		for(EventMention event1: list) {
			//System.out.println(Utils.getText(event1.getTreeNode()));
			alreadyConsidered.add(event1);
			for(EventMention event2: list) {
				if(!alreadyConsidered.contains(event2)) {
					String type = Utils.getEventEventRelation(example.gold, event1.getTreeNode(), event2.getTreeNode()).toString();
					BioDatum newDatum = new BioDatum(null, Utils.getText(event1.getTreeNode()) + "-" + Utils.getText(event2.getTreeNode()), type, event1, event2);
					newDatum.features = computeFeatures(example, event1, event2);
					newData.add(newDatum);
				}
		    }
		}
    	return newData;
	}

}
