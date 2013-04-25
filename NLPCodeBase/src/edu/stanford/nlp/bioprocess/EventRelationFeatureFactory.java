package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.ArgumentRelation.EventType;
import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;

public class EventRelationFeatureFactory {

	private boolean printAnnotations = false, printDebug = false;
	private boolean useLexicalFeatures;
	HashMap<String, String> verbForms = Utils.getVerbForms();
	List<String> TemporalConnectives = Arrays.asList(new String[]{"before", "after", "since","when", "meanwhile", "lately", 
										"then", "subsequently", "previously", "next", "later", "subsequent", "previous"});

	public EventRelationFeatureFactory(boolean useLexicalFeatures) {
		this.useLexicalFeatures = useLexicalFeatures;
		// TODO Auto-generated constructor stub
	}
	
	private FeatureVector computeFeatures(Example example, EventMention event1, EventMention event2) {
		List<String> features = new ArrayList<String>();
		CoreLabel event1CoreLabel = Utils.findCoreLabelFromTree(event1.getSentence(), event1.getTreeNode()),
				event2CoreLabel = Utils.findCoreLabelFromTree(event2.getSentence(), event2.getTreeNode());
		
		//Is event2 immediately after event1?
		features.add("isImmediatelyAfter:" + Utils.isEventNextInOrder(example.gold.get(EventMentionsAnnotation.class), event1, event2));
		
		//Add words in between two event mentions.
		List<Pair<String, String>> wordsInBetween = Utils.findWordsInBetween(example, event1, event2);
		for(int wordCounter = 0; wordCounter < wordsInBetween.size(); wordCounter++) {
			//Ignore if it is a noun
			//if(!word.second().startsWith("NN"))
			String POS = wordsInBetween.get(wordCounter).second, word = wordsInBetween.get(wordCounter).first;
			String POS2 = wordCounter < wordsInBetween.size() - 1? wordsInBetween.get(wordCounter + 1).second : "", 
					word2 =  wordCounter < wordsInBetween.size() - 1? wordsInBetween.get(wordCounter + 1).first : "";
			if(POS.startsWith("VB") && POS2.equals("IN")) { 
				features.add("wordsInBetween:" + word + " " + word2);
				wordCounter++;
			}
			else
				features.add("wordsInBetween:" + word);
			if(TemporalConnectives.contains(word.toLowerCase())) {
				features.add("temporalConnective:" + word.toLowerCase());
			}
		}
		
		//POS tags of both events
		features.add("POS:" + event1.getTreeNode().value() + "+" + event2.getTreeNode().value());
		
		//Lemmas of both events
		String lemma1 = event1CoreLabel.lemma().toLowerCase();
		if(verbForms.containsKey(lemma1)) {
			lemma1 = verbForms.get(lemma1);
		}
		String lemma2 = event2CoreLabel.lemma().toLowerCase();
		if(verbForms.containsKey(lemma2)) {
			lemma2 = verbForms.get(lemma2);
		}
		features.add("Lemma:" + lemma1 + "+" + lemma2);
		
		//Are the lemmas same?
		features.add("eventLemmasSame:" + lemma1.equals(lemma2));
		
		//Number of sentences and words between two event mentions. Quantized to 'Low', 'Medium', 'High' etc.
		Pair<Integer, Integer> counts =  Utils.findNumberOfSentencesAndWordsBetween(example, event1, event2);
		features.add("numSentencesInBetween:" + quantizedSentenceCount(counts.first()));
		features.add("numWordsInBetween:" + quantizedWordCount(counts.second()));
		
		//Features if the two triggers are in the same sentence.
		if (counts.first() == 0) {
			//Lowest common ancestor between the two event triggers. Reduces score.
			Tree root = event1.getSentence().get(TreeCoreAnnotations.TreeAnnotation.class);
			Tree lca = Trees.getLowestCommonAncestor(event1.getTreeNode(), event2.getTreeNode(), root);
			features.add("lowestCommonAncestor:" + lca.value());
			
			
			Tree node = lca;
			
			while(!node.value().equals("ROOT")) {
				node = node.parent(root);
			}
			
			//Are the event triggers part of a Prepositional phrase individually (one feature each)?
			boolean matched = false;
			node = event1.getTreeNode();
			root = event1.getSentence().get(TreeCoreAnnotations.TreeAnnotation.class);
			while(!node.value().equals("ROOT")) {
				if(node.value().equals("PP")) {
					for(Tree ponode:node.postOrderNodeList()) {
						//System.out.println(ponode);
						if(ponode.isPreTerminal() && (ponode.value().equals("IN") || ponode.value().equals("TO"))) {
							//System.out.println("added feature!!");
							features.add("1partOfPP:" + ponode.firstChild().value());
							matched = true;
							break;
						}
					}
				}
				
				if(matched)
					break;
				node = node.parent(root);
			}
			
			matched = false;
			node = event2.getTreeNode();
			root = event2.getSentence().get(TreeCoreAnnotations.TreeAnnotation.class);
			while(!node.value().equals("ROOT")) {
				if(node.value().equals("PP")) {
					//System.out.println("Found PP!!");
					for(Tree ponode:node.postOrderNodeList()) {
						if(ponode.isPreTerminal() && (ponode.value().equals("IN") || ponode.value().equals("TO"))) {
							//System.out.println("added feature!!");
							features.add("2partOfPP:" + ponode.firstChild().value());
							matched = true;
							break;
						}
					}
				}
				if(matched)
					break;
				node = node.parent(root);
			}
			
			//Dependency path if the event triggers are in the same sentence.
			String deppath2to1 = Utils.getDependencyPath(event1.getSentence(), event1.getTreeNode(), event2.getTreeNode());
			String deppath1to2 = Utils.getDependencyPath(event1.getSentence(), event2.getTreeNode(), event1.getTreeNode());
			if(!deppath1to2.isEmpty()) {
				features.add("deppath1to2:" + deppath1to2 );
				//Does event1 dominate event2
				features.add("1dominates2");
			}
			if(!deppath2to1.isEmpty()) {
				features.add("deppath2to1:" + deppath2to1 );
				features.add("2dominates1");
			}
		}
		
		//Finding "mark" relationship
		SemanticGraph graph1 = event1.getSentence().get(CollapsedCCProcessedDependenciesAnnotation.class);
		IndexedWord indexedWord1 = Utils.findDependencyNode(event1.getSentence(), event1.getTreeNode());
		for(SemanticGraphEdge e: graph1.getOutEdgesSorted(indexedWord1)) {
			if(e.getRelation().getShortName().equals("mark")) {
				features.add("markRelationEvent1:" + e.getTarget().originalText());
			}
			if(e.getRelation().getShortName().equals("advmod")) {
				features.add("advmodRelationEvent1:" + e.getTarget().originalText());
			}
		}
		SemanticGraph graph2 = event2.getSentence().get(CollapsedCCProcessedDependenciesAnnotation.class);
		IndexedWord indexedWord2 = Utils.findDependencyNode(event2.getSentence(), event2.getTreeNode());
		for(SemanticGraphEdge e: graph2.getOutEdgesSorted(indexedWord2)) {
			if(e.getRelation().getShortName().equals("mark")) {
				features.add("markRelationEvent2:" + e.getTarget().originalText());
			}
			if(e.getRelation().getShortName().equals("advmod")) {
				features.add("advmodRelationEvent2:" + e.getTarget().originalText());
			}
		}
		
		//Do they share a common argument in the dependency tree?
		if (counts.first() == 0) {
			List<SemanticGraphEdge> edges1 = graph1.getOutEdgesSorted(indexedWord1);
			List<SemanticGraphEdge> edges2 = graph2.getOutEdgesSorted(indexedWord2);
			System.out.println("Trying semgraph " + event1.getTreeNode() + ":" + event2.getTreeNode());
			for(SemanticGraphEdge e1:edges1) {
				for(SemanticGraphEdge e2:edges2) {
					if(e1.getTarget() == e2.getTarget()) {
						System.out.println("Share a child" + example.id);
						break;
					}
				}
			}
		}
		
		//Word, lemma and POS before first event and word after second event. Dummy words added if first word or last word respectively.
		//Lemma and POS are not good features.
		CoreLabel word1Before = Utils.findWordBefore(event1, 1), word1After = Utils.findWordAfter(event2, 1),
				word2Before = Utils.findWordBefore(event1, 2), word2After = Utils.findWordAfter(event2, 2);

		if(word1Before!=null) {
			String word = word1Before.originalText();
			features.add("word1BeforeFirstEvent:" + word);
			//features.add("POS1BeforeFirstEvent:" + word1Before.get(PartOfSpeechAnnotation.class));
			//features.add("lemma1BeforeFirstEvent:" + (verbForms.containsKey(word)?verbForms.get(word):word));
		}
		else {
			features.add("word1BeforeFirstEvent:" + "DUMMY_WORD");
			//features.add("POS1BeforeFirstEvent:" + "DUMMY_POS");
			//features.add("lemma1BeforeFirstEvent:" + "DUMMY_LEMMA");
		}
		if(word1After!=null) {	
			String word = word1After.originalText();
			features.add("word1AfterSecondEvent:" + word);
			//features.add("POS1AfterSecondEvent:" + word1After.get(PartOfSpeechAnnotation.class));
			//features.add("lemma1AfterSecondEvent:" + (verbForms.containsKey(word)?verbForms.get(word):word));
		}
		else {
			features.add("word1AfterSecondEvent:" + "DUMMY_WORD");
			//features.add("POS1AfterSecondEvent:" + "DUMMY_POS");
			//features.add("lemma1AfterSecondEvent:" + "DUMMY_LEMMA");
		}
		if(word2Before!=null) {
			String word = word2Before.originalText();
			features.add("word2BeforeFirstEvent:" + word);
			//features.add("POS2BeforeFirstEvent:" + word2Before.get(PartOfSpeechAnnotation.class));
			//features.add("lemma2BeforeFirstEvent:" + (verbForms.containsKey(word)?verbForms.get(word):word));
		}
		else {
			features.add("word2BeforeFirstEvent:" + "DUMMY_WORD");
			//features.add("POS2BeforeFirstEvent:" + "DUMMY_POS");
			//features.add("lemma2BeforeFirstEvent:" + "DUMMY_LEMMA");
		}
		if(word2After!=null) {
			String word = word2After.originalText();
			features.add("word2AfterSecondEvent:" + word);
			//features.add("POS2AfterSecondEvent:" + word2After.get(PartOfSpeechAnnotation.class));
			//features.add("lemma2AfterSecondEvent:" + (verbForms.containsKey(word)?verbForms.get(word):word));
		}
		else {
			features.add("word2AfterSecondEvent:" + "DUMMY_WORD");
			//features.add("POS2AfterSecondEvent:" + "DUMMY_POS");
			//features.add("lemma2AfterSecondEvent:" + "DUMMY_LEMMA");
		}

		//POS bigram of event trigger and the word before. Makes result slightly worse
		if(word1Before != null) {
			//features.add("POSbigram:" + event1CoreLabel.get(PartOfSpeechAnnotation.class) + "+" + word1Before.get(PartOfSpeechAnnotation.class));
		}
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

	public String quantizedSentenceCount(int numSentences) {
		if(numSentences == 0) {
			return "None";
		}
		else if(numSentences == 1) {
			return "Low";
		}
		else if(numSentences == 2 || numSentences == 3) {
			return "Medium";
		}
		return "High";
	}
	
	public String quantizedWordCount(int numWords) {
		if(numWords <= 3) {
			return "Low";
		}
		else if(numWords <= 7) {
			return "Medium";
		}
		else if(numWords <= 15) {
			return "High";
		}
		return "VeryHigh";
	}
}
