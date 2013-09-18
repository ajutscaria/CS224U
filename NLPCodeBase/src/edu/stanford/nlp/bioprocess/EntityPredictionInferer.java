package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;

import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;

public class EntityPredictionInferer extends Inferer {
	private boolean printDebugInformation = false;
	List<BioDatum> prediction = null;
	
	public EntityPredictionInferer() {
		
	}
	
	public EntityPredictionInferer(List<BioDatum> predictions){
		this.prediction = predictions;
	}

	public List<BioDatum> BaselineInfer(List<Example> examples, Params parameters, FeatureExtractor ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>();
		//EntityFeatureFactory ff = new EntityFeatureFactory();
		for(Example example:examples) {
			LogInfo.begin_track("Example %s",example.id);
			
			for(CoreMap sentence: example.gold.get(SentencesAnnotation.class)) {
				Set<Tree> eventNodes = null;
				if(prediction == null)
					eventNodes = Utils.getEventNodesFromSentence(sentence).keySet();
				else
					eventNodes = Utils.getEventNodesForSentenceFromDatum(prediction, sentence);
				List<BioDatum> test = ff.setFeaturesTest(sentence, eventNodes, example.id);
				for(Tree event:eventNodes) {
					LogInfo.logs("******************Event " + Utils.getText(event)+ 
								"[" + (Utils.getEventNodesFromSentence(sentence).containsKey(event)?"Correct":"Wrong") +"]**********************");
					List<BioDatum> testDataEvent = new ArrayList<BioDatum>();
					for(BioDatum d:test)
						if(d.eventNode == event) {
							testDataEvent.add(d);
						}
					
					for(BioDatum d:testDataEvent) {
						if(d.entityNode.value().equals("NP") && Utils.isNodesRelated(sentence, d.entityNode, event))
							d.guessLabel = "E";
						else
							d.guessLabel = "O";
					}
					predicted.addAll(testDataEvent);
					
					LogInfo.logs(sentence);
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
			}
			//LogInfo.end_track();
		}
		return predicted;
	}
	
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
				if(prediction == null)
					eventNodes = Utils.getEventNodesFromSentence(sentence).keySet();
				else
					eventNodes = Utils.getEventNodesForSentenceFromDatum(prediction, sentence);
				List<BioDatum> test = ff.setFeaturesTest(sentence, eventNodes, ex.id);
				
				for(Tree event:eventNodes) {
					//LogInfo.logs("******************Event " + Utils.getText(event)+ 
					//		"[" + (Utils.getEventNodesFromSentence(sentence).containsKey(event)?"Correct":"Wrong") +"]**********************");
					List<BioDatum> testDataEvent = new ArrayList<BioDatum>();
					for(BioDatum d:test)
						if(d.eventNode == event) {
							//LogInfo.logs(d.entityNode);
							testDataEvent.add(d);
						}
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
					
					predicted.addAll(testDataEvent);
					
					//LogInfo.logs(sentence);
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
				for(EventMention ev:sentence.get(EventMentionsAnnotation.class))
					if(!eventNodes.contains(ev.getTreeNode())){
						LogInfo.logs("||||||||||||||||||||||Event " +ev.getTreeNode()+ "[Missed]||||||||||||||");
						LogInfo.logs("\n---------Missed entities-------------------------");
						for(ArgumentRelation m:ev.getArguments())
							LogInfo.logs(m.mention.getTreeNode());
						LogInfo.logs("------------------------------------------\n");
					}
			}
			//LogInfo.end_track();
		}
		return predicted;
	}
	
}
