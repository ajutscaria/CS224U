package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import fig.basic.LogInfo;

public class EventPredictionInferer extends Inferer {
	boolean printDebugInformation = true;
	List<BioDatum> prediction = null;
	public static double eventRecall = 0;
	
	public EventPredictionInferer() {
		
	}
	
	public EventPredictionInferer(List<BioDatum> predictions) {
		prediction = predictions;
	}
	
	public List<BioDatum> baselineInfer(List<Example> examples, Params parameters, FeatureExtractor ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>(); 
		//EventFeatureFactory ff = new EventFeatureFactory();
		for(Example example:examples) {
			for(CoreMap sentence: example.gold.get(SentencesAnnotation.class)) {
				List<BioDatum> test = ff.setFeaturesTest(sentence, Utils.getEntityNodesFromSentence(sentence), example.id);
				
				for(BioDatum d:test)
					if(d.entityNode.isPreTerminal() && d.entityNode.value().startsWith("VB"))
						d.guessLabel = "E";
					else
						d.guessLabel = "O";
				predicted.addAll(test);
			}
		}
		return predicted;
	}
	
	public List<BioDatum> infer(List<Example> testData, Params parameters, FeatureExtractor ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>(); 
		for(Example ex:testData) {
			//LogInfo.begin_track("Example %s",ex.id);

			for(CoreMap sentence:ex.gold.get(SentencesAnnotation.class)) {
				Set<Tree> entityNodes = null;
				if(prediction == null)
					entityNodes = Utils.getEntityNodesFromSentence(sentence);
				else
					entityNodes = Utils.getEntityNodesForSentenceFromDatum(prediction, sentence);
				
				LinearClassifier<String, String> classifier = new LinearClassifier<String, String>(parameters.weights, parameters.featureIndex, parameters.labelIndex);
				List<BioDatum> dataset = ff.setFeaturesTest(sentence, entityNodes, ex.id);

				for(BioDatum d:dataset) {
					Datum<String, String> newDatum = new BasicDatum<String, String>(d.getFeatures(),d.label());
					d.setPredictedLabel(classifier.classOf(newDatum));
					
					//@ heather 
					double scoreE = classifier.scoreOf(newDatum, "E"), scoreO = classifier.scoreOf(newDatum, "O");
					d.setEventProbability(Math.exp(scoreE)/(Math.exp(scoreE) + Math.exp(scoreO)));
					 
				}
				predicted.addAll(dataset);
				if(printDebugInformation) {
					LogInfo.logs(sentence);
					LogInfo.logs(sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennString());
					LogInfo.logs(sentence.get(CollapsedCCProcessedDependenciesAnnotation.class));

					LogInfo.logs("\n---------GOLD EVENTS-------------------------");
					for(EventMention m:sentence.get(EventMentionsAnnotation.class)) 
							LogInfo.logs(m.getTreeNode());
					
					LogInfo.logs("---------PREDICTIONS-------------------------");
					for(BioDatum d:dataset)
						if(d.predictedLabel().equals("E") || d.label().equals("E"))
							LogInfo.logs(String.format("%-30s Gold:  %s Predicted: %s", d.word, d.label(), d.predictedLabel()));
					LogInfo.logs("------------------------------------------\n");
				}
			}
			
			//LogInfo.end_track();
		}
		return predicted;
	}
	
	public List<BioDatum> inferilp(List<Example> testData, Params parameters, FeatureExtractor ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>();
		double theta = Main.theta;
		int counter = 0;
		double globaleventLossRecall = 0;
		for(Example ex:testData) {
			int eventLost = 0;
			//LogInfo.begin_track("Example %s",ex.id);
			List<BioDatum> oneprocess = new ArrayList<BioDatum>();
			ex.prediction.set(EntityMentionsAnnotation.class, new ArrayList<EntityMention>());
			ex.prediction.set(EventMentionsAnnotation.class, new ArrayList<EventMention>());
			for(CoreMap sentence:ex.gold.get(SentencesAnnotation.class)) {
				Set<Tree> entityNodes = null;
				System.out.println("\n"+sentence);
				/*if(prediction == null)
					entityNodes = Utils.getEntityNodesFromSentence(sentence);
				else
					entityNodes = Utils.getEntityNodesForSentenceFromDatum(prediction, sentence);*/
				
				LinearClassifier<String, String> classifier = new LinearClassifier<String, String>(parameters.weights, parameters.featureIndex, parameters.labelIndex);
				List<BioDatum> dataset = ff.setFeaturesTest(sentence, entityNodes, ex.id);

				for(BioDatum d:dataset) {
					Datum<String, String> newDatum = new BasicDatum<String, String>(d.getFeatures(),d.label());
					d.setPredictedLabel(classifier.classOf(newDatum));
					
					//@ heather 
					double scoreE = classifier.scoreOf(newDatum, "E"), scoreO = classifier.scoreOf(newDatum, "O");
					scoreE = (Math.exp(scoreE)/(Math.exp(scoreE) + Math.exp(scoreO)));
					d.setEventProbability(scoreE);
					EventMention m;
					if(scoreE >= theta){
						predicted.add(d);
						m = new EventMention("", sentence, null);
						m.setTreeNode(d.eventNode);
						m.setProb(scoreE);
						//System.out.println("evenmention size:"+ex.prediction.get(EventMentionsAnnotation.class).size());
						Utils.addAnnotation(ex.prediction, (EventMention)m, false);
						//System.out.println("evenmention size:"+ex.prediction.get(EventMentionsAnnotation.class).size());
						//for(EventMention em:ex.prediction.get(EventMentionsAnnotation.class))
						//	System.out.println(em.getTreeNode().toString());
					} else{
						if(d.label.equals("E"))eventLost++;
						counter++;
					}
					 
				}
				//predicted.addAll(dataset);
				printDebugInformation = false;
				if(printDebugInformation) {
					LogInfo.logs(sentence);
					LogInfo.logs(sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennString());
					LogInfo.logs(sentence.get(CollapsedCCProcessedDependenciesAnnotation.class));

					//JONATHAN _ commented out since seems redundant with PREDICTIONS that also prints gold labels
//					LogInfo.logs("\n---------GOLD EVENTS-------------------------");
//					for(EventMention m:sentence.get(EventMentionsAnnotation.class)) 
//							LogInfo.logs(m.getTreeNode());
					
					LogInfo.logs("---------PREDICTIONS-------------------------");
					for(BioDatum d:dataset)
						if(d.predictedLabel().equals("E") || d.label().equals("E"))
							LogInfo.logs(String.format("%-30s Gold=%s, Predicted=%s, predictedProb=%s", d.word, d.label(), d.predictedLabel(),d.probEvent));
					LogInfo.logs("------------------------------------------\n");
				}
			}
			
			//LogInfo.end_track();
			int goldevents = ex.gold.get(EventMentionsAnnotation.class).size();
			double recall = (double)(goldevents-eventLost)/goldevents;
			LogInfo.logs("Events lost:"+eventLost+", Total gold events:"+goldevents
					+", Recall:"+recall);
			globaleventLossRecall += recall;
		}
		LogInfo.logs("Average Recall:"+globaleventLossRecall/testData.size());
		eventRecall += globaleventLossRecall/testData.size();
		System.out.println("Not added due to theta: "+counter);
		return predicted;
	}
}
