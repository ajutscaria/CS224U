package edu.stanford.nlp.bioprocess;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
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
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.OneToOneMap.OneToOneMapException;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.wsd.WordNet.WordNetID;
import fig.basic.LogInfo;

public class EventRelationFeatureFactory {

	private boolean printAnnotations = false, printDebug = false;
	public static Set<String> markWords = new HashSet<String>(), advmodWords = new HashSet<String>(), eventInsidePP = new HashSet<String>();
	private boolean useLexicalFeatures;
	//WnExpander wnLexicon;
	HashMap<String, String> verbForms = Utils.getVerbForms();
	List<String> TemporalConnectives = Arrays.asList(new String[]{"before", "after", "since","when", "meanwhile", "lately", 
										"then", "subsequently", "previously", "next", "later", "subsequent", "previous"});
	HashMap<String, Integer> clusters = Utils.loadClustering();

	public EventRelationFeatureFactory(boolean useLexicalFeatures) {
		this.useLexicalFeatures = useLexicalFeatures;
		/*try {
			wnLexicon = new WnExpander();
		} catch (IOException | OneToOneMapException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
		
		// TODO Auto-generated constructor stub
	}
	
	private FeatureVector computeFeatures(Example example, EventMention event1, EventMention event2) {
		List<String> features = new ArrayList<String>();
		CoreLabel event1CoreLabel = Utils.findCoreLabelFromTree(event1.getSentence(), event1.getTreeNode()),
				event2CoreLabel = Utils.findCoreLabelFromTree(event2.getSentence(), event2.getTreeNode());
		boolean isImmediatelyAfter = Utils.isEventNextInOrder(example.gold.get(EventMentionsAnnotation.class), event1, event2),
				isAfter = Utils.isEventNext(example.gold.get(EventMentionsAnnotation.class), event1, event2);
		//Is event2 immediately after event1?
		if(Main.features.contains("isImmediatelyAfter"))
			features.add("isImmediatelyAfter:" + isImmediatelyAfter);
		
		features.add("isAfter:" + isAfter);
		
		//Add words in between two event mentions if they are adjacent in text.
		List<Pair<String, String>> wordsInBetween = Utils.findWordsInBetween(example, event1, event2);
		if(isImmediatelyAfter) {
			for(int wordCounter = 0; wordCounter < wordsInBetween.size(); wordCounter++) {
				//Ignore if it is a noun - NOT GOOD
				//if(!wordsInBetween.get(wordCounter).second().startsWith("NN")) {
					String POS = wordsInBetween.get(wordCounter).second, word = wordsInBetween.get(wordCounter).first;
					String POS2 = wordCounter < wordsInBetween.size() - 1? wordsInBetween.get(wordCounter + 1).second : "", 
							word2 =  wordCounter < wordsInBetween.size() - 1? wordsInBetween.get(wordCounter + 1).first : "";
					if(POS.startsWith("VB") && POS2.equals("IN")) { 
						if(Main.features.contains("wordsInBetween"))
							features.add("wordsInBetween:" + word + " " + word2);
						wordCounter++;
					}
					else if(Main.features.contains("wordsInBetween"))
						features.add("wordsInBetween:" + word);
					if(TemporalConnectives.contains(word.toLowerCase())) {
						if(Main.features.contains("temporalConnective"))
							features.add("temporalConnective:" + word.toLowerCase());
					}
				//}
			}
		}
		//Is there an and within 5 words of each other
		if(wordsInBetween.size() <= 5) {
			for(int wordCounter = 0; wordCounter < wordsInBetween.size(); wordCounter++) {
				if(wordsInBetween.get(wordCounter).first.equals("and")) 
					features.add("closeAndInBetween");
			}
		}
		
		String pos1 = event1.getTreeNode().value(), pos2 = event2.getTreeNode().value();
		//POS tags of both events
		if(Main.features.contains("POS"))
			features.add("POS:" + pos1 + "+" + pos2);
		
		//If event1 is the first event in the paragraph and is a nominalization, it is likely that others are sub-events
		if(example.gold.get(EventMentionsAnnotation.class).indexOf(event1) == 0 && event1.getTreeNode().value().startsWith("NN")) {
			//features.add("firstAndNominalization");
		}
		
		//Lemmas of both events
		String lemma1 = event1CoreLabel.lemma().toLowerCase();
		if(verbForms.containsKey(lemma1)) {
			lemma1 = verbForms.get(lemma1);
		}
		String lemma2 = event2CoreLabel.lemma().toLowerCase();
		if(verbForms.containsKey(lemma2)) {
			lemma2 = verbForms.get(lemma2);
		}
		if(Main.features.contains("lemma"))
			features.add("lemmas:" + lemma1 + "+" + lemma2);
		
		//Number of sentences and words between two event mentions. Quantized to 'Low', 'Medium', 'High' etc.
		Pair<Integer, Integer> counts =  Utils.findNumberOfSentencesAndWordsBetween(example, event1, event2);
		if(Main.features.contains("numSentencesInBetween"))
			features.add("numSentencesInBetween:" + quantizedSentenceCount(counts.first()));
		if(Main.features.contains("numWordsInBetween"))
			features.add("numWordsInBetween:" + quantizedWordCount(counts.second()));
		
		
		//String poss = event1.getTreeNode().value() + "+" + event2.getTreeNode().value();
		//Are the lemmas same?
		if(Main.features.contains("eventLemmasSame"))
			//if(!poss.equals("VBZ+VBZ") && !poss.equals("VBN+VBN"))
		{	
			features.add("eventLemmasSame:" + lemma1.equals(lemma2));
			//features.add("eventLemmasSame:" + lemma1.equals(lemma2) + (pos1.startsWith("NN") || pos2.startsWith("NN") ||
			//		pos1.equals("VBG") || pos2.equals("VBG")));
		}
		
		//If second trigger is noun, the determiner related to it in dependency tree.
		if(pos2.startsWith("NN")) {
			String determiner = Utils.getDeterminer(event2.getSentence(), event2.getTreeNode());
			if(determiner != null) {
				features.add("determinerBefore2:" + determiner);
				//LogInfo.logs("determiner:" + determiner);
			}
		}
		
		//Features if the two triggers are in the same sentence.
		if (counts.first() == 0) {
			//Lowest common ancestor between the two event triggers. Reduces score.
			Tree root = event1.getSentence().get(TreeCoreAnnotations.TreeAnnotation.class);
			Tree lca = Trees.getLowestCommonAncestor(event1.getTreeNode(), event2.getTreeNode(), root);
			if(Main.features.contains("lowestCommonAncestor"))
				features.add("lowestCommonAncestor:" + lca.value());
			
			//Dependency path if the event triggers are in the same sentence.
			String deppath2to1 = Utils.getUndirectedDependencyPath(event1.getSentence(), event1.getTreeNode(), event2.getTreeNode());
			String deppath1to2 = Utils.getUndirectedDependencyPath(event1.getSentence(), event2.getTreeNode(), event1.getTreeNode());
			if(!deppath1to2.isEmpty()) {
				if(Main.features.contains("deppath1to2")) {
					features.add("deppath1to2:" + deppath1to2 );
				}
				//Does event1 dominate event2
				if(Main.features.contains("1dominates2"))
					features.add("1dominates2");
			}
			if(!deppath2to1.isEmpty()) {
				if(Main.features.contains("deppath2to1")) {
					features.add("deppath2to1:" + deppath2to1 );
				}
				if(Main.features.contains("2dominates1"))
					features.add("2dominates1");
			}
		}
		
	
		//Finding "mark" and "advmod" relationship
		SemanticGraph graph1 = event1.getSentence().get(CollapsedCCProcessedDependenciesAnnotation.class);
		IndexedWord indexedWord1 = Utils.findDependencyNode(event1.getSentence(), event1.getTreeNode());
		for(SemanticGraphEdge e: graph1.getOutEdgesSorted(indexedWord1)) {
			if(e.getRelation().getShortName().equals("mark")) {
				if(Main.features.contains("markRelationEvent1")&& counts.first == 0) {
					features.add("markRelationEvent1:" + e.getTarget().originalText());
					markWords.add(e.getTarget().originalText().toLowerCase());
				}
			}
			if(e.getRelation().getShortName().equals("advmod")) {
				if(Main.features.contains("advmodRelationEvent1")) {
					features.add("advmodRelationEvent1:" + e.getTarget().originalText());
					advmodWords.add(e.getTarget().originalText().toLowerCase());
				}
			}
		}
		SemanticGraph graph2 = event2.getSentence().get(CollapsedCCProcessedDependenciesAnnotation.class);
		IndexedWord indexedWord2 = Utils.findDependencyNode(event2.getSentence(), event2.getTreeNode());
		for(SemanticGraphEdge e: graph2.getOutEdgesSorted(indexedWord2)) {
			if(e.getRelation().getShortName().equals("mark")) {
				if(Main.features.contains("markRelationEvent2")&& counts.first == 0) {
					features.add("markRelationEvent2:" + e.getTarget().originalText());
					markWords.add(e.getTarget().originalText().toLowerCase());		
				}
			}
			if(e.getRelation().getShortName().equals("advmod")) {
				if(Main.features.contains("advmodRelationEvent2")) {
					features.add("advmodRelationEvent2:" + e.getTarget().originalText());
					advmodWords.add(e.getTarget().originalText().toLowerCase());
				}
			}
		}
		
		//See if the two triggers share a common lemma as child in the dependency graph.
		List<Pair<IndexedWord, String>> w1Children = new ArrayList<Pair<IndexedWord, String>>();
		for(SemanticGraphEdge e: graph1.getOutEdgesSorted(indexedWord1)) {
			w1Children.add(new Pair<IndexedWord, String>(e.getTarget() , e.getRelation().toString()));
		}
		
		for(SemanticGraphEdge e: graph2.getOutEdgesSorted(indexedWord2)) {
			for(Pair<IndexedWord, String> pair: w1Children) {
				if(e.getTarget().originalText().equals(pair.first.originalText())) {
					//System.out.println(indexedWord1 + ":" + indexedWord2 + " share children. " + pair.first.originalText()
					//		 +":" +pair.second+ "+" +e.getRelation().toString());
					//features.add("shareSameLemmaAsChild:"+ pair.second+ "+" +e.getRelation().toString());
				}
				//System.out.println(indexedWord1 + ":" + indexedWord2 + " share children. " + w.originalText());
				//features.add("shareSameLemmaAsChild")// + w.originalText());
			}
		}
		
		//Do they share a common argument in the dependency tree? (if they are in the same sentence)
		if (counts.first() == 0) {
			List<SemanticGraphEdge> edges1 = graph1.getOutEdgesSorted(indexedWord1);
			List<SemanticGraphEdge> edges2 = graph2.getOutEdgesSorted(indexedWord2);
			for(SemanticGraphEdge e1:edges1) {
				for(SemanticGraphEdge e2:edges2) {
					if(e1.getTarget().equals(e2.getTarget())) {
						if(Main.features.contains("shareChild"))
							features.add("shareChild:" + e1.getRelation() + "+" + e2.getRelation());
						break;
					}
				}
			}
		}

		
		if(isImmediatelyAfter) {
			Tree root = event1.getSentence().get(TreeCoreAnnotations.TreeAnnotation.class);
			Tree node = event1.getTreeNode();
			
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
						if(ponode.isPreTerminal() && (ponode.value().equals("IN") || ponode.value().equals("TO"))) {
							if(Main.features.contains("1partOfPP")) {
								features.add("1partOfPP:" + ponode.firstChild().value());
								eventInsidePP.add(ponode.firstChild().value().toLowerCase());
							}
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
					for(Tree ponode:node.postOrderNodeList()) {
						if(ponode.isPreTerminal() && (ponode.value().equals("IN") || ponode.value().equals("TO"))) {
							if(Main.features.contains("2partOfPP")) {
								features.add("2partOfPP:" + ponode.firstChild().value());
								eventInsidePP.add(ponode.firstChild().value().toLowerCase());
							}
							matched = true;
							break;
						}
					}
				}
				if(matched)
					break;
				node = node.parent(root);
			}
		}
		
		//Get first word in the sentence - Not Good
		//features.add("firstWord1" + event1.getSentence().get(TokensAnnotation.class).get(0).originalText());
		//features.add("firstWord2" + event2.getSentence().get(TokensAnnotation.class).get(0).originalText());
		
		//WordNet Synsets?
		/*Set<WordNetID> set1 = wnLexicon.getSynsets(Utils.getText(event1.getTreeNode()), event1.getTreeNode().value());
		Set<WordNetID> set2 = wnLexicon.getSynsets(Utils.getText(event2.getTreeNode()), event2.getTreeNode().value());
		if(set1!=null && set2!=null) {
			boolean synsetMatch = false;
			for(WordNetID w1:set1) {
				for(WordNetID w2:set2){
					if(w1.equals(w2)) {
						features.add("synsetsOfEachOther");
						synsetMatch = true;
						break;
					}
				}
				if(synsetMatch)
					break;
			}
		}*/
		
		//Word, lemma and POS before first event and word after second event. Dummy words added if first word or last word respectively.
		//Lemma and POS are not good features. - NOT USEFUL
		CoreLabel word1Before = Utils.findWordBefore(event1, 1), word1After = Utils.findWordAfter(event2, 1),
				word2Before = Utils.findWordBefore(event1, 2), word2After = Utils.findWordAfter(event2, 2);

		if(word1Before!=null) {
			String word = word1Before.originalText();
			//features.add("word1BeforeFirstEvent:" + word);
			//features.add("POS1BeforeFirstEvent:" + word1Before.get(PartOfSpeechAnnotation.class));
			//features.add("lemma1BeforeFirstEvent:" + (verbForms.containsKey(word)?verbForms.get(word):word));
		}
		else {
			//features.add("word1BeforeFirstEvent:" + "DUMMY_WORD");
			//features.add("POS1BeforeFirstEvent:" + "DUMMY_POS");
			//features.add("lemma1BeforeFirstEvent:" + "DUMMY_LEMMA");
		}
		if(word1After!=null) {	
			String word = word1After.originalText();
			//features.add("word1AfterSecondEvent:" + word);
			//features.add("POS1AfterSecondEvent:" + word1After.get(PartOfSpeechAnnotation.class));
			//features.add("lemma1AfterSecondEvent:" + (verbForms.containsKey(word)?verbForms.get(word):word));
		}
		else {
			//features.add("word1AfterSecondEvent:" + "DUMMY_WORD");
			//features.add("POS1AfterSecondEvent:" + "DUMMY_POS");
			//features.add("lemma1AfterSecondEvent:" + "DUMMY_LEMMA");
		}
		if(word2Before!=null) {
			String word = word2Before.originalText();
			//features.add("word2BeforeFirstEvent:" + word);
			//features.add("POS2BeforeFirstEvent:" + word2Before.get(PartOfSpeechAnnotation.class));
			//features.add("lemma2BeforeFirstEvent:" + (verbForms.containsKey(word)?verbForms.get(word):word));
		}
		else {
			//features.add("word2BeforeFirstEvent:" + "DUMMY_WORD");
			//features.add("POS2BeforeFirstEvent:" + "DUMMY_POS");
			//features.add("lemma2BeforeFirstEvent:" + "DUMMY_LEMMA");
		}
		if(word2After!=null) {
			String word = word2After.originalText();
			//features.add("word2AfterSecondEvent:" + word);
			//features.add("POS2AfterSecondEvent:" + word2After.get(PartOfSpeechAnnotation.class));
			//features.add("lemma2AfterSecondEvent:" + (verbForms.containsKey(word)?verbForms.get(word):word));
		}
		else {
			//features.add("word2AfterSecondEvent:" + "DUMMY_WORD");
			//features.add("POS2AfterSecondEvent:" + "DUMMY_POS");
			//features.add("lemma2AfterSecondEvent:" + "DUMMY_LEMMA");
		}

		//POS bigram of event trigger and the word before. Makes result slightly worse
		if(word1Before != null) {
			//features.add("POSbigram:" + event1CoreLabel.get(PartOfSpeechAnnotation.class) + "+" + word1Before.get(PartOfSpeechAnnotation.class));
		}
		features.add("bias");
		
		FeatureVector fv = new FeatureVector(features);
		if(printDebug) {
			LogInfo.logs(String.format("\nExample : %s Event 1 - %s, Event 2 - %s", example.id, event1.getTreeNode(), event2.getTreeNode()));
			LogInfo.logs(fv.getFeatureString());
			LogInfo.logs("---------------------------------------------------------\n");
		}
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
			alreadyConsidered.add(event1);
			for(EventMention event2: list) {
				if(!alreadyConsidered.contains(event2)) {
					String type = Utils.getEventEventRelation(example.gold, event1.getTreeNode(), event2.getTreeNode()).toString();
					BioDatum newDatum = new BioDatum(null, Utils.getText(event1.getTreeNode()) + "-" + Utils.getText(event2.getTreeNode()), type, event1, event2);
					newDatum.features = computeFeatures(example, event1, event2);
					if(newDatum.features.getFeatures().contains(("eventLemmasSame:truetrue")))
							LogInfo.logs("eventLemmasSame:" + newDatum.event1.getTreeNode() + ":" + newDatum.event2.getTreeNode() + ":" +newDatum.label);
					newDatum.setExampleID(example.id);
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
		else if(numWords <= 6) {
			return "Medium";
		}
		else if(numWords <= 15) {
			return "High";
		}
		return "VeryHigh";
	}
}
