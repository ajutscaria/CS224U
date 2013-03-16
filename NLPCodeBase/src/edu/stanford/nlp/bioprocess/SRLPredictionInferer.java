package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;

public class SRLPredictionInferer extends Inferer {
	private boolean printDebugInformation = false;
	List<BioDatum> prediction = null;
	
	public SRLPredictionInferer() {
		
	}
	
	public SRLPredictionInferer(List<BioDatum> predictions){
		this.prediction = predictions;
	}

	public List<BioDatum> BaselineInfer(List<Example> examples, Params parameters, FeatureExtractor ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>();
		String popularRelation = ((SRLFeatureFactory)ff).getMostCommonRelationType();
		//EntityFeatureFactory ff = new EntityFeatureFactory();
		for(Example example:examples) {
			LogInfo.begin_track("Example %s",example.id);
			
			for(CoreMap sentence: example.gold.get(SentencesAnnotation.class)) {
				Set<Tree> eventNodes = null;
				if(prediction == null)
					eventNodes = Utils.getEventNodesFromSentence(sentence).keySet();
				else
					eventNodes = Utils.getEventNodesForSentenceFromDatum(prediction, sentence);
				List<BioDatum> test = ff.setFeaturesTest(sentence, eventNodes);
				for(Tree event:eventNodes) {
					LogInfo.logs("------------------Event " + Utils.getText(event)+"--------------");
					List<BioDatum> testDataEvent = new ArrayList<BioDatum>();
					for(BioDatum d:test)
						if(d.eventNode == event) {
							testDataEvent.add(d);
						}
					List<BioDatum> testDataWithLabel = new ArrayList<BioDatum>();
	
					for (int i = 0; i < testDataEvent.size(); i += parameters.getLabelIndex().size()) {
						testDataWithLabel.add(testDataEvent.get(i));
					}
					
					for(BioDatum d:testDataWithLabel) {
						if((d.entityNode.value().equals("NP") /*|| d.entityNode.value().startsWith("NN")*/) && Utils.isNodesRelated(sentence, d.entityNode, event)) {
							d.guessRole = popularRelation;
						}
						else
							d.guessRole = RelationType.NONE.toString();
					}
					predicted.addAll(testDataWithLabel);
					
					LogInfo.logs(sentence);
					//sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
					
					LogInfo.logs("\n---------GOLD ENTITIES-------------------------");
					for(BioDatum d:testDataWithLabel) 
						if(!d.role.equals(RelationType.NONE.toString()))
							LogInfo.logs(d.entityNode + ":" + d.role);
					
					LogInfo.logs("---------PREDICTIONS-------------------------");
					for(BioDatum d:testDataWithLabel)
					{
						if(!(d.guessRole.equals(RelationType.NONE.toString()) && d.role.equals(RelationType.NONE.toString()))) {
							LogInfo.logs(String.format("%-30s [%s], Gold:  %s Predicted: %s", d.word, d.entityNode.getSpan(), d.role, d.guessRole));
						}
					}
					LogInfo.logs("------------------------------------------\n");
				}
			}
			LogInfo.end_track();
		}
		return predicted;
	}
	
	public List<BioDatum> Infer(List<Example> testData, Params parameters, FeatureExtractor ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>();
		//EntityFeatureFactory ff = new EntityFeatureFactory();
		for(Example ex:testData) {
			LogInfo.begin_track("Example %s",ex.id);
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
				List<BioDatum> test = ff.setFeaturesTest(sentence, eventNodes);
				
				for(Tree event:eventNodes) {
					LogInfo.logs("------------------Event: " + Utils.getText(event)+"--------------");
					List<BioDatum> testDataEvent = new ArrayList<BioDatum>();
					for(BioDatum d:test)
						if(d.eventNode == event) {
							//LogInfo.logs(d.entityNode);
							testDataEvent.add(d);
						}
					List<BioDatum> testDataWithLabel = new ArrayList<BioDatum>();
					
					LinearClassifier<String, String> classifier = new LinearClassifier<>(parameters.weights, parameters.featureIndex, parameters.labelIndex);
					
					for(BioDatum d:testDataEvent) {
						Datum<String, String> newDatum = new BasicDatum<String, String>(d.getFeatures(),d.label());
						d.setPredictedLabel(classifier.classOf(newDatum));
						double scoreE = classifier.scoreOf(newDatum, "E"), scoreO = classifier.scoreOf(newDatum, "O");
						d.setProbability(Math.exp(scoreE)/(Math.exp(scoreE) + Math.exp(scoreO)));
						//LogInfo.logs(d.word + ":" + d.predictedLabel() + ":" + d.getProbability());
					}
	
					IdentityHashMap<Tree, List<Pair<String, Double>>> map = new IdentityHashMap<Tree, List<Pair<String, Double>>>();
	
					for(BioDatum d:testDataWithLabel) {
						if (Utils.subsumesEvent(d.entityNode, sentence)) {
							List<Pair<String, Double>> blankList = new ArrayList<Pair<String, Double>>();
							for (int i=0; i<parameters.getLabelIndex().size(); i++) {
								if (i == 0) {
									blankList.add(new Pair<String, Double>((String)parameters.getLabelIndex().get(i), 1.0));
								} else {
									blankList.add(new Pair<String, Double>((String)parameters.getLabelIndex().get(i), 0.0));
								}
							}
							map.put(d.entityNode, blankList);
						} else {
							map.put(d.entityNode, d.getRankedRoles());
						}
					}
					
					//DynamicProgrammingSRL dynamicProgrammerSRL = new DynamicProgrammingSRL(sentence, map, testDataWithLabel, parameters.getLabelIndex());
					//dynamicProgrammerSRL.calculateLabels();
					
					predicted.addAll(testDataWithLabel);
					
					LogInfo.logs("\n---------GOLD ENTITIES-------------------------");
					for(BioDatum d:testDataWithLabel) 
						if(!d.role.equals(RelationType.NONE.toString()))
							LogInfo.logs(d.entityNode + ":" + d.role);
					
					LogInfo.logs("---------PREDICTIONS-------------------------");
					for(BioDatum d:testDataWithLabel)
					{
						if(!(d.guessRole.equals(RelationType.NONE.toString()) && d.role.equals(RelationType.NONE.toString()))) {
							LogInfo.logs(String.format("%-30s [%s], Gold:  %s Predicted: %s", d.word, d.entityNode.getSpan(), d.role, d.guessRole));
						}
					}
					LogInfo.logs("------------------------------------------\n");
				}
			}
			LogInfo.end_track();

		}
		return predicted;
	}
	
}
