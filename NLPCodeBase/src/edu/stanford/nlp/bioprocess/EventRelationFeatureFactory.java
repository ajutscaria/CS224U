package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;

public class EventRelationFeatureFactory {

	private boolean printAnnotations = false, printDebug = false;
	public static Set<String> markWords = new HashSet<String>(), advmodWords = new HashSet<String>(), eventInsidePP = new HashSet<String>();
	private boolean useBaselineFeaturesOnly = false, runGlobalModel = false;
	private String model = "";
	HashMap<String, String> verbForms = Utils.getVerbForms();
	List<String> TemporalConnectives = Arrays.asList(new String[]{"before", "after", "since", "when", "meanwhile", "lately", 
									"include","includes","including","included", "first", "begin","begins","began","beginning","begun","start","starts","started","starting",
									"lead","leads","causes","cause","result","results",
									"then", "subsequently", "previously", "next", "later", "subsequent", "previous"});

	List<String> diffClauseRelations = Arrays.asList(new String[]{"acomp", "advcl", "ccomp", "csubj", "infmod", "prepc", "purpcl", "xcomp"});
	HashMap<String, String> MarkAndPPClusters = new HashMap<String, String>();
	HashMap<String, String> AdvModClusters = new HashMap<String, String>();
	
	HashMap<String, Integer> clusters = Utils.loadClustering();

	public EventRelationFeatureFactory(boolean useLexicalFeatures, String model) {
		this.model = model;
		if(model.equals("localbase")) {
			useBaselineFeaturesOnly = true;
		}
		else if(model.equals("global")) {
			runGlobalModel = true;
		}
		MarkAndPPClusters.put("if", RelationType.PreviousEvent.toString());
		MarkAndPPClusters.put("until", RelationType.NextEvent.toString());
		MarkAndPPClusters.put("after", RelationType.PreviousEvent.toString());
		MarkAndPPClusters.put("as", RelationType.CotemporalEvent.toString());
		MarkAndPPClusters.put("because", RelationType.Causes.toString());
		MarkAndPPClusters.put("before", RelationType.NextEvent.toString());
		MarkAndPPClusters.put("since", RelationType.Causes.toString());
		MarkAndPPClusters.put("so", RelationType.Caused.toString());
		MarkAndPPClusters.put("while", RelationType.CotemporalEvent.toString());
		MarkAndPPClusters.put("during", RelationType.SuperEvent.toString());
		MarkAndPPClusters.put("upon", RelationType.PreviousEvent.toString());
		
		AdvModClusters.put("then", RelationType.PreviousEvent.toString());
		AdvModClusters.put("thus", RelationType.Causes.toString());
		AdvModClusters.put("also", RelationType.CotemporalEvent.toString());
		AdvModClusters.put("eventually", RelationType.PreviousEvent.toString());
		AdvModClusters.put("meanwhile", RelationType.CotemporalEvent.toString());
		AdvModClusters.put("thereby", RelationType.Causes.toString());
		AdvModClusters.put("finally", RelationType.PreviousEvent.toString());
		AdvModClusters.put("first", RelationType.SuperEvent.toString());
		AdvModClusters.put("hence", RelationType.Causes.toString());
		AdvModClusters.put("later", RelationType.PreviousEvent.toString());
		AdvModClusters.put("next", RelationType.PreviousEvent.toString());
		AdvModClusters.put("simultaneously", RelationType.CotemporalEvent.toString());
		AdvModClusters.put("subsequently", RelationType.PreviousEvent.toString());
		AdvModClusters.put("if", RelationType.NextEvent.toString());
		AdvModClusters.put("until", RelationType.PreviousEvent.toString());
		AdvModClusters.put("after", RelationType.NextEvent.toString());
		AdvModClusters.put("as", RelationType.CotemporalEvent.toString());
		AdvModClusters.put("because", RelationType.Caused.toString());
		AdvModClusters.put("so", RelationType.Causes.toString());
		AdvModClusters.put("result", RelationType.Causes.toString());
		AdvModClusters.put("results", RelationType.Causes.toString());
		AdvModClusters.put("lead", RelationType.Causes.toString());
		AdvModClusters.put("leads", RelationType.Causes.toString());
		AdvModClusters.put("cause", RelationType.Causes.toString());
		AdvModClusters.put("causes", RelationType.Causes.toString());
		AdvModClusters.put("while", RelationType.CotemporalEvent.toString());
		AdvModClusters.put("during", RelationType.SubEvent.toString());
		AdvModClusters.put("upon", RelationType.NextEvent.toString());
		AdvModClusters.put("include", RelationType.SuperEvent.toString());
		AdvModClusters.put("includes", RelationType.SuperEvent.toString());
		AdvModClusters.put("included", RelationType.SuperEvent.toString());
		AdvModClusters.put("including", RelationType.SuperEvent.toString());
		AdvModClusters.put("begin", RelationType.SuperEvent.toString());
		AdvModClusters.put("begins", RelationType.SuperEvent.toString());
		AdvModClusters.put("began", RelationType.SuperEvent.toString());
		AdvModClusters.put("begun", RelationType.SuperEvent.toString());
		AdvModClusters.put("beginning", RelationType.SuperEvent.toString());
		AdvModClusters.put("start", RelationType.SuperEvent.toString());
		AdvModClusters.put("starts", RelationType.SuperEvent.toString());
		AdvModClusters.put("started", RelationType.SuperEvent.toString());
		AdvModClusters.put("starting", RelationType.SuperEvent.toString());
		AdvModClusters.put("subsequent", RelationType.PreviousEvent.toString());
		AdvModClusters.put("previously", RelationType.NextEvent.toString());
		AdvModClusters.put("previous", RelationType.NextEvent.toString());
		
		/*try {
			wnLexicon = new WnExpander();
		} catch (IOException | OneToOneMapException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/		
	}
	
	private FeatureVector computeFeatures(Example example, List<EventMention> mentions, EventMention event1, EventMention event2) {
		List<String> features = new ArrayList<String>();
		CoreLabel event1CoreLabel = Utils.findCoreLabelFromTree(event1.getSentence(), event1.getTreeNode()),
				event2CoreLabel = Utils.findCoreLabelFromTree(event2.getSentence(), event2.getTreeNode());
		boolean isImmediatelyAfter = Utils.isEventNextInOrder(mentions, event1, event2);
		List<Pair<String, String>> wordsInBetween = Utils.findWordsInBetween(example, event1, event2);
		//Number of sentences and words between two event mentions. Quantized to 'Low', 'Medium', 'High' etc.
		Pair<Integer, Integer> countsSentenceAndWord =  Utils.findNumberOfSentencesAndWordsBetween(example, event1, event2);
		int sentenceBetweenEvents = countsSentenceAndWord.first();
		int wordsBetweenEvents = countsSentenceAndWord.second();
		
		SemanticGraph graph1 = event1.getSentence().get(CollapsedCCProcessedDependenciesAnnotation.class);
		IndexedWord indexedWord1 = Utils.findDependencyNode(event1.getSentence(), event1.getTreeNode());
		SemanticGraph graph2 = event2.getSentence().get(CollapsedCCProcessedDependenciesAnnotation.class);
		IndexedWord indexedWord2 = Utils.findDependencyNode(event2.getSentence(), event2.getTreeNode());
		
		String pos1 = event1.getTreeNode().value(), pos2 = event2.getTreeNode().value();
		
		//Lemmas of both events
		String lemma1 = event1CoreLabel.lemma().toLowerCase();
		if(verbForms.containsKey(lemma1)) {
			lemma1 = verbForms.get(lemma1);
		}
		String lemma2 = event2CoreLabel.lemma().toLowerCase();
		if(verbForms.containsKey(lemma2)) {
			lemma2 = verbForms.get(lemma2);
		}
		features.add("lemmas:" + lemma1 + "+" + lemma2);
		
		//Is event2 immediately after event1?
		if(!runGlobalModel) {
			features.add("isImmediatelyAfter:" + isImmediatelyAfter);
		}
		
		if(isImmediatelyAfter) {
			//Add words in between two event mentions if they are adjacent in text.
			StringBuffer phrase = new StringBuffer();
			for(int wordCounter = 0; wordCounter < wordsInBetween.size(); wordCounter++) {
				phrase.append(wordsInBetween.get(wordCounter).first + " ");
				String POS = wordsInBetween.get(wordCounter).second, word = wordsInBetween.get(wordCounter).first;
				String POS2 = wordCounter < wordsInBetween.size() - 1? wordsInBetween.get(wordCounter + 1).second : "", 
						word2 =  wordCounter < wordsInBetween.size() - 1? wordsInBetween.get(wordCounter + 1).first : "";
						
				if(!TemporalConnectives.contains(word.toLowerCase())) {
					if(POS.startsWith("VB") && POS2.equals("IN")) { 
						features.add("wordsInBetween:" + word + " " + word2);
						wordCounter++;
					}
					else
						features.add("wordsInBetween:" + word);
				}
				else {	
					if(sentenceBetweenEvents < 2) {
						//LogInfo.logs("TEMPORAL CONNECTIVE ADDED: " + example.id + " " + lemma1 + " " + lemma2 + " " + word.toLowerCase());
						if(useBaselineFeaturesOnly) {
							features.add("temporalConnective:" + word.toLowerCase());
						}
						else {
							features.add("connector:" + word.toLowerCase());
							if(AdvModClusters.containsKey(word.toLowerCase())) {
								features.add("connectorCluster:" + AdvModClusters.get(word.toLowerCase()));
							}
						}
					}
				}
			}


		}
		
		if(!useBaselineFeaturesOnly) {
			if(isImmediatelyAfter) {
				//Is there an and within 5 words of each other
				if(wordsInBetween.size() <= 5) {
					for(int wordCounter = 0; wordCounter < wordsInBetween.size(); wordCounter++) {
						if(wordsInBetween.get(wordCounter).first.equals("and")) 
								features.add("closeAndInBetween");
					}
				}
				
				//If event1 is the first event in the paragraph and is a nominalization, it is likely that others are sub-events
				if(mentions.indexOf(event1) == 0 && event1.getTreeNode().value().startsWith("NN")) {
					features.add("firstAndNominalization");
				}
			}
			//Are the lemmas same?	
			features.add("eventLemmasSame:" + lemma1.equals(lemma2));
			
			//If second trigger is noun, the determiner related to it in dependency tree.
			if(pos2.startsWith("NN")) {
				String determiner = Utils.getDeterminer(event2.getSentence(), event2.getTreeNode());
				if(determiner != null) {
					features.add("determinerBefore2:" + determiner);
				}
			}
		}
		
		//POS tags of both events
		features.add("POS:" + pos1 + "+" + pos2);
		features.add("numSentencesInBetween:" + quantizedSentenceCount(sentenceBetweenEvents));
		features.add("numWordsInBetween:" + quantizedWordCount(wordsBetweenEvents));
		
		//Features if the two triggers are in the same sentence.
		if (sentenceBetweenEvents == 0) {
			//Lowest common ancestor between the two event triggers. Reduces score.
			Tree root = event1.getSentence().get(TreeCoreAnnotations.TreeAnnotation.class);
			Tree lca = Trees.getLowestCommonAncestor(event1.getTreeNode(), event2.getTreeNode(), root);
			features.add("lowestCommonAncestor:" + lca.value());
			
			//Dependency path if the event triggers are in the same sentence.
			//LogInfo.logs(example.id + " " + lemma1 + " " + lemma2);
			String deppath = Utils.getUndirectedDependencyPath_Events(event1.getSentence(), event1.getTreeNode(), event2.getTreeNode());
			if(!deppath.isEmpty()) {
				if(!useBaselineFeaturesOnly) {
					features.add("deppath:" + deppath);
					features.add("deppathwithword:" + Utils.getUndirectedDependencyPath_Events_WithWords(event1.getSentence(), event1.getTreeNode(), event2.getTreeNode()));
				}
				//Does event1 dominate event2
				if(deppath.contains("->") && !deppath.contains("<-")) {
					features.add("1dominates2");
				}
				
				if(deppath.contains("<-") && !deppath.contains("->")) {
					features.add("2dominates1");
				}
			}
			
			//Extract mark relation
		    //LogInfo.logs("Trying Marker: " + example.id + " " + lemma1 + " " + lemma2 + " ");
			List<Pair<String, String>> markRelations = extractMarkRelation(mentions, event1, event2);
			for(Pair<String, String> markRelation: markRelations) {
				//LogInfo.logs("MARKER ADDED: " + example.id + " " + lemma1 + " " + lemma2 + " " + markRelation);
				if(useBaselineFeaturesOnly) {
					features.add("markRelation:" + markRelation.first());
				}
				else {
					features.add("connector:" + markRelation.first());
					//In some cases, we don't have clusters for some relation.
					if(!markRelation.second().isEmpty())
						features.add("connectorCluster:" + markRelation.second());
				}
			}
				
			//Extract PP relation
			//LogInfo.logs("Trying PP: " + example.id + " " + lemma1 + " " + lemma2 + " ");
			List<Pair<String, String>> ppRelations = extractPPRelation(mentions, event1, event2);
			for(Pair<String, String> ppRelation: ppRelations) {
				//LogInfo.logs("PP ADDED: " + example.id + " " + lemma1 + " " + lemma2 + " " + ppRelation);
				if(useBaselineFeaturesOnly) {
					features.add("PPRelation:" + ppRelation.first());
				}
				else {
					features.add("connector:" + ppRelation.first());
					//In some cases, we don't have clusters (if we haven't included in the list.
					if(!ppRelation.second().isEmpty()) {
						features.add("connectorCluster:" + ppRelation.second());
					}
				}
			}
		}
		
		if(isImmediatelyAfter) {
			//Extract advmod relation
			//LogInfo.logs("Trying AdvMod: " + example.id + " " + lemma1 + " " + lemma2 + " ");
			List<Pair<String, String>> advModRelations = extractAdvModRelation(mentions, event1, event2);
			for(Pair<String, String> advModRelation: advModRelations) {
				//LogInfo.logs("ADVMOD ADDED: " + example.id + " " + lemma1 + " " + lemma2 + " " + advModRelation);
				if(useBaselineFeaturesOnly) {
					features.add("advModRelation:" + advModRelation.first());
				}
				else {
					features.add("connector:" + advModRelation.first());
					//In some cases, we don't have clusters for some relation.
					if(!advModRelation.second().isEmpty()) {
						features.add("connectorCluster:" + advModRelation.second());
					}
				}
			}
		}
		
		String advMod = extractAdvModRelation(graph2, indexedWord2);
		if(advMod != null && !advMod.isEmpty()) {
			features.add("advMod:" + advMod);
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
		if (sentenceBetweenEvents == 0) {
			List<SemanticGraphEdge> edges1 = graph1.getOutEdgesSorted(indexedWord1);
			List<SemanticGraphEdge> edges2 = graph2.getOutEdgesSorted(indexedWord2);
			for(SemanticGraphEdge e1:edges1) {
				for(SemanticGraphEdge e2:edges2) {
					if(e1.getTarget().equals(e2.getTarget())) {
						features.add("shareChild:" + e1.getRelation() + "+" + e2.getRelation());
						break;
					}
				}
			}
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
						newDatum.features = computeFeatures(ex, ex.gold.get(EventMentionsAnnotation.class), event1, event2);
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
					newDatum.features = computeFeatures(example, list, event1, event2);
					
					newDatum.setExampleID(example.id);
					newData.add(newDatum);
				}
		    }
		}
    	return newData;
	}
	
	private List<Pair<String, String>> extractMarkRelation(List<EventMention> mentions, EventMention eventMention1, EventMention eventMention2){
		CoreMap sentence = eventMention1.getSentence();
		Tree event1 = eventMention1.getTreeNode(), event2 = eventMention2.getTreeNode();
		SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
		IndexedWord indexedWord1 = Utils.findDependencyNode(sentence, event1), indexedWord2 = Utils.findDependencyNode(sentence, event2) ;
		int event1Index = indexedWord1.index(), event2Index = indexedWord2.index();
		
		List<Pair<String, String>> markRelations = new ArrayList<Pair<String,String>>();
		
		Pair<String, String> markRelation1 = extractMarkRelation(mentions, graph, eventMention1,
																					indexedWord1, indexedWord2, event1Index, event2Index);
		Pair<String, String> markRelation2 = extractMarkRelation(mentions, graph, eventMention1,
																					indexedWord2, indexedWord1, event1Index, event2Index);
		if(markRelation1 != null) {
			markRelations.add(new Pair<String, String>(markRelation1.first() + "_1", markRelation1.second()));
		}
		
		if(markRelation2 != null) {
			markRelations.add(new Pair<String, String>(markRelation2.first() + "_2", markRelation2.second()));
		}
		
		return markRelations;
	}
	
	private Pair<String, String> extractMarkRelation(List<EventMention> mentions, SemanticGraph graph, EventMention eventMention1, 
																		IndexedWord indexedWordThis, IndexedWord indexedWordThat, int event1Index, int event2Index) {
		for(SemanticGraphEdge e: graph.getOutEdgesSorted(indexedWordThis)) {
			if(e.getRelation().getShortName().equals("mark")) {
				int markIndex = e.getTarget().index();
				String markerName = e.getTarget().lemma().toLowerCase();
				
				//Check if the source of incoming edge (advcl, ccmod, dep) is same as indexedWordThat or parent of indexedWordThat
				if (graph.getIncomingEdgesSorted(indexedWordThis).size() > 0) {
					SemanticGraphEdge edge = graph.getIncomingEdgesSorted(indexedWordThis).get(0);
							
					IndexedWord parent = edge.getSource();
	
					if(parent.index() == indexedWordThat.index() || isInSameDependencyClauseAndChild(graph, parent, indexedWordThat)) {
						if(Utils.isFirstEventInSentence(mentions, eventMention1) && markIndex < event1Index) {
							//LogInfo.logs("Marker before :" +markerName);
							if(MarkAndPPClusters.containsKey(markerName))
								return new Pair<String, String>(markerName, MarkAndPPClusters.get(markerName));
							else
								return new Pair<String, String>(markerName, "");
						}
						else if(markIndex < event2Index) {
							//LogInfo.logs("Marker between :" +markerName);
							if(MarkAndPPClusters.containsKey(markerName))
								return new Pair<String, String>(markerName, Utils.getInverseRelation(MarkAndPPClusters.get(markerName)));
							else
								return new Pair<String, String>(markerName, "");
						}
						else {
							LogInfo.logs("Marker after");
						}
					}
				}
			}
		}
		return null;
	}
	
	private boolean isInSameDependencyClauseAndChild(SemanticGraph graph, IndexedWord parent, IndexedWord word) {
		List<SemanticGraphEdge> edges = graph.getShortestDirectedPathEdges(parent, word);
		if(edges == null)
			return false;

		for(SemanticGraphEdge edge:edges) {
			if(diffClauseRelations.contains(edge.getRelation().getShortName()))
				return false;
		}
		return true;
	}
	
	private List<Pair<String, String>> extractAdvModRelation(List<EventMention> mentions, EventMention eventMention1, EventMention eventMention2) {
		
		CoreMap sentence1 = eventMention1.getSentence();
		CoreMap sentence2 = eventMention2.getSentence();
		Tree event1 = eventMention1.getTreeNode(), event2 = eventMention2.getTreeNode();
		SemanticGraph graph1 = sentence1.get(CollapsedCCProcessedDependenciesAnnotation.class);
		SemanticGraph graph2 = sentence2.get(CollapsedCCProcessedDependenciesAnnotation.class);
		IndexedWord indexedWord1 = Utils.findDependencyNode(sentence1, event1), indexedWord2 = Utils.findDependencyNode(sentence2, event2) ;
		int event1Index = indexedWord1.index(), event2Index = indexedWord2.index();

		List<Pair<String, String>> advModRelations = new ArrayList<Pair<String,String>>();
		
		Pair<String, String> advModRelation1 = extractAdvModRelation(mentions, graph1, eventMention1,
				indexedWord1, indexedWord2, event1Index, event2Index);
		Pair<String, String> advModRelation2 = extractAdvModRelation(mentions, graph2, eventMention1,
						indexedWord2, indexedWord1, event1Index, event2Index);
		if(advModRelation1 != null) {
			//advModRelations.add(new Pair<String, String>(advModRelation1.first() + "_1", advModRelation1.second()));
		}
		
		if(advModRelation2 != null) {
			advModRelations.add(new Pair<String, String>(advModRelation2.first(), advModRelation2.second()));
		}
		
		return advModRelations;
	}
	
	private Pair<String, String> extractAdvModRelation(List<EventMention> mentions,
			SemanticGraph graph, EventMention eventMention1,
			IndexedWord indexedWordThis, IndexedWord indexedWordThat,
			int event1Index, int event2Index) {
		for(SemanticGraphEdge e: graph.getOutEdgesSorted(indexedWordThis)) {
			if(e.getRelation().getShortName().equals("advmod")) {
				String advModName = e.getTarget().lemma().toLowerCase();
				
				//LogInfo.logs("AdvMod found '" + indexedWordThis + "':" + advModName);
				if(AdvModClusters.containsKey(advModName))
					return new Pair<String, String>(advModName, AdvModClusters.get(advModName));
				else
					return new Pair<String, String>(advModName, "");
			}
		}
		return null;
	}
	
	private String extractAdvModRelation(SemanticGraph graph, IndexedWord indexedWord) {
		for(SemanticGraphEdge e: graph.getOutEdgesSorted(indexedWord)) {
			if(e.getRelation().getShortName().equals("advmod")) {
				return e.getTarget().originalText();
			}
		}
		return null;
	}

	private List<Pair<String, String>> extractPPRelation(List<EventMention> mentions, EventMention eventMention1, EventMention eventMention2){
		CoreMap sentence = eventMention1.getSentence();
		Tree event1 = eventMention1.getTreeNode(), event2 = eventMention2.getTreeNode();
		Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		
		int event1Index = event1.nodeNumber(root), event2Index = event2.nodeNumber(root);
		//Are the event triggers part of a Prepositional phrase individually (one feature each)?
		
		List<Pair<String, String>> ppRelations = new ArrayList<Pair<String,String>>();
		
		Pair<String, String> ppRelation1 = extractPPRelation(mentions, eventMention1, event1,
																					event2, event1Index, event2Index);
		Pair<String, String> ppRelation2 = extractPPRelation(mentions, eventMention1, event2,
																					event1, event1Index, event2Index);
		if(ppRelation1 != null) {
			ppRelations.add(new Pair<String, String>(ppRelation1.first() + "_1", ppRelation1.second()));
		}
		
		if(ppRelation2 != null) {
			ppRelations.add(new Pair<String, String>(ppRelation2.first() + "_2", ppRelation2.second()));
		}
		
		return ppRelations;
	}
	
	private Pair<String, String> extractPPRelation(List<EventMention> mentions, EventMention eventMention1, Tree thisEvent, Tree thatEvent, 																							
																							int event1Index, int event2Index) {
		Tree root = eventMention1.getSentence().get(TreeCoreAnnotations.TreeAnnotation.class);
		Tree node = thisEvent;
		//root.pennPrint();
		//Are the event triggers part of a Prepositional phrase individually (one feature each)?
		while(!node.value().equals("ROOT") && !node.value().equals("S") && !node.value().equals("SBAR")) {
			if(node.value().equals("PP")) {
				for(Tree ponode:node.postOrderNodeList()) {
					if(ponode.isPreTerminal() && (ponode.value().equals("IN") || ponode.value().equals("TO"))) {
						//Other event should not be in the same PP
						if(node.dominates(thatEvent)) {
							//LogInfo.logs("PP dominates other event");
							return null;
						}
						Tree lca = Trees.getLowestCommonAncestor(thisEvent, thatEvent, root);
						//LogInfo.logs("ANCESTOR " + lca.value());
						List<String> path = Trees.pathNodeToNode(lca, ponode, lca);
						path.remove(path.size()-1);
						path.remove(0);
						path.remove(0);

						//LogInfo.logs("PPPATH " + thisEvent + " " + path);
						//if(path.contains("down-S") || path.contains("down-SBAR")) {
						if(path.contains("down-S") || path.contains("down-SBAR")) {
							//LogInfo.logs("Contains S or SBAR");
							//return null;
						}
						
						path = Trees.pathNodeToNode(lca, thatEvent, lca);
						path.remove(path.size()-1);
						path.remove(0);
						path.remove(0);

						//LogInfo.logs("PPPATH " + thatEvent + " " + path);
						if(path.contains("down-S") || path.contains("down-SBAR")) {
							//LogInfo.logs("Contains S or SBAR");
							//return null;
						}
						
						String ppName = ponode.firstChild().value().toLowerCase();
						int ppIndex = ponode.firstChild().nodeNumber(root);
						if(Utils.isFirstEventInSentence(mentions, eventMention1) && ppIndex < event1Index) {
							//LogInfo.logs("PP before :" +ppName);
							if(MarkAndPPClusters.containsKey(ppName))
								return new Pair<String, String>(ppName, MarkAndPPClusters.get(ppName));
							else
								return new Pair<String, String>(ppName, "");
						}
						else if(ppIndex > event1Index && ppIndex < event2Index) {
							//LogInfo.logs("PP between :" +ppName);
							if(MarkAndPPClusters.containsKey(ppName))
								return new Pair<String, String>(ppName, Utils.getInverseRelation(MarkAndPPClusters.get(ppName)));
							else
								return new Pair<String, String>(ppName, "");
						}
						else if(ppIndex > event2Index){
							//LogInfo.logs("PP after");
						}
						
						break;
					}
				}
			}
			node = node.parent(root);
		}
		return null;
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
		if(numWords <= 4) {
			return "Low";
		}
		else if(numWords <= 8) {
			return "Medium";
		}
		else if(numWords <= 15) {
			return "High";
		}
		return "VeryHigh";
	}
}
