package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;

public class EntityStandaloneInferer extends Inferer{
	private boolean printDebugInformation = false, useRule = false;
	Set<String> nominalizations = Utils.getNominalizedVerbs();
	@Override
	public List<BioDatum> BaselineInfer(List<Example> examples, Params parameters,
			FeatureExtractor ff) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<BioDatum> Infer(List<Example> testData, Params parameters, FeatureExtractor ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>();
		//EntityFeatureFactory ff = new EntityFeatureFactory();
		for(Example ex:testData) {
			//LogInfo.begin_track("Example %s",ex.id);
			//IdentityHashSet<Tree> entities = Utils.getEntityNodes(ex);
			
			for(CoreMap sentence:ex.gold.get(SentencesAnnotation.class)) {
				if(printDebugInformation ) {
					LogInfo.logs(sentence);
					LogInfo.logs(sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennString());
					LogInfo.logs(sentence.get(CollapsedCCProcessedDependenciesAnnotation.class));
				}
				Set<Tree> eventNodes = null;

				List<BioDatum> test = ff.setFeaturesTest(sentence, eventNodes, ex.id);
				
				List<BioDatum> testDataEvent = new ArrayList<BioDatum>();
				for(BioDatum d:test)
					testDataEvent.add(d);
				
				if(!useRule) {
					LinearClassifier<String, String> classifier = new LinearClassifier<String, String>(parameters.weights, parameters.featureIndex, parameters.labelIndex);
					
					for(BioDatum d:testDataEvent) {
						Datum<String, String> newDatum = new BasicDatum<String, String>(d.getFeatures(),d.label());
						d.setPredictedLabel(classifier.classOf(newDatum));
						double scoreE = classifier.scoreOf(newDatum, "E"), scoreO = classifier.scoreOf(newDatum, "O");
						d.setProbability(Math.exp(scoreE)/(Math.exp(scoreE) + Math.exp(scoreO)));
						//LogInfo.logs(d.word + ":" + d.predictedLabel() + ":" + d.getProbability());
					}
					IdentityHashMap<Tree, Pair<Double, String>> map = new IdentityHashMap<Tree, Pair<Double, String>>();
					for(BioDatum d:testDataEvent) {
						if (Utils.subsumesEvent(d.entityNode, sentence)) {
							map.put(d.entityNode, new Pair<Double, String>(0.0, "O"));
						} else {
							map.put(d.entityNode, new Pair<Double, String>(d.getProbability(), d.guessLabel));
						}
					}
					
					DynamicProgramming dynamicProgrammer = new DynamicProgramming(sentence, map, testDataEvent);
					dynamicProgrammer.calculateLabels();
				}
				
				//My own inferer
				else {
					//IdentityHashSet<Tree> entities = new IdentityHashSet<Tree>();
					for(BioDatum d:testDataEvent) {
						Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
						
						CoreLabel label = Utils.findCoreLabelFromTree(sentence, d.entityNode);
						//LogInfo.logs(d.word + ":" +label.originalText());
						//0.442, 0.774, 0.562
						//if(d.entityNode.value().equals("NP") && checkIfEntity(sentence, d.entityNode)) {
						//0.358, 0.953, 0.520
						//if(d.entityNode.value().equals("NP")) {
						//0.501, 0.718, 0.588
						if(d.entityNode.value().equals("NP") 
								&& !nominalizations.contains(label.originalText()) 
								&& d.entityNode.getLeaves().size() < 8 
								&& checkIfEntity(sentence, d.entityNode)
								&& !d.entityNode.getLeaves().get(0).value().equals("Figure")
								&& !root.getLeaves().get(0).value().equals("-LRB-")
								//&& !(d.entityNode.getLeaves().size() == 1 && (label.get(PartOfSpeechAnnotation.class).equals("CD") || label.equals("this")))
								//&& !Utils.getText(d.entityNode).contains(" in a ")
								//&& !Utils.getText(d.entityNode).contains(" result")
								//&& !Utils.getText(d.entityNode).contains(" and ")
								) 
						{
							boolean noParentPresent = true;
							//for(Tree entityNodesEarlier:entities)
							//	if(entityNodesEarlier.dominates(d.entityNode))
							//		noParentPresent = false;
							if(noParentPresent) {
								//entities.add(d.entityNode);
								d.guessLabel = "E";
							}
							//else {
							//	d.guessLabel = "O";
							//}
						}
						else {
							d.guessLabel = "O";
						}
					}
				}
				predicted.addAll(testDataEvent);
				
					//sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
				if(printDebugInformation) {
					LogInfo.logs("\n---------GOLD ENTITIES-------------------------");
					for(BioDatum d:testDataEvent) 
						if(d.label.equals("E"))
							LogInfo.logs(d.entityNode + ":" + d.label);
					
					LogInfo.logs("---------PREDICTIONS-------------------------");
					for(BioDatum d:testDataEvent)
						if(d.guessLabel.equals("E") || d.label.equals("E"))
							LogInfo.logs(String.format("%-30s [%s], Gold:  %s Predicted: %s", d.word, d.entityNode.getSpan(), d.label, d.guessLabel));
					LogInfo.logs("------------------------------------------\n");
				}
			}
			
			//LogInfo.end_track();
		}
		return predicted;
	}
	
	boolean hasNPchild(Tree node) {
		for(Tree child:node.postOrderNodeList()) {
			if(child == node || child.isPreTerminal() || child.isLeaf())
				continue;
			if(child.value().equals("NP"))
				return true;
		}
		return false;
	}
	
	private boolean checkContainsNominalizations(CoreMap sentence, Tree node) {
		for(int i = node.getSpan().getSource(); i <= node.getSpan().getTarget();i++) {
			if(nominalizations.contains(sentence.get(TokensAnnotation.class).get(i).originalText()))
					return true;
		}
		return false;
	}
	
    private boolean checkIfEntity(CoreMap sentence, Tree node) {
    	IndexedWord word = Utils.findDependencyNode(sentence, node);
    	SemanticGraph graph = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
    	for(SemanticGraphEdge e: graph.getIncomingEdgesSorted(word)) {
    		String POSParent = e.getSource().toString().split("-")[1], parent = e.getSource().toString().split("-")[0];
    		if(POSParent.startsWith("VB") || nominalizations.contains(parent))
    			return true;
    	}
    	return false;
    }
}
