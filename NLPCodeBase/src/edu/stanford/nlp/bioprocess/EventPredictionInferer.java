package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import fig.basic.LogInfo;

public class EventPredictionInferer extends Inferer {
	boolean printDebugInformation = true;
	List<BioDatum> prediction = null;
	
	public EventPredictionInferer() {
		
	}
	
	public EventPredictionInferer(List<BioDatum> predictions) {
		prediction = predictions;
	}
	
	public List<BioDatum> BaselineInfer(List<Example> examples, Params parameters, FeatureExtractor ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>(); 
		//EventFeatureFactory ff = new EventFeatureFactory();
		for(Example example:examples) {
			for(CoreMap sentence: example.gold.get(SentencesAnnotation.class)) {
				List<BioDatum> test = ff.setFeaturesTest(sentence, Utils.getEntityNodesFromSentence(sentence));
				List<BioDatum> testDataWithLabel = new ArrayList<BioDatum>();

				for (int i = 0; i < test.size(); i += parameters.getLabelIndex().size()) {
					testDataWithLabel.add(test.get(i));
				}
				
				for(BioDatum d:testDataWithLabel)
					if(d.entityNode.isPreTerminal() && d.entityNode.value().startsWith("VB"))
						d.guessLabel = "E";
					else
						d.guessLabel = "O";
				predicted.addAll(testDataWithLabel);
			}
		}
		return predicted;
	}
	
	public List<BioDatum> Infer(List<Example> testData, Params parameters, FeatureExtractor ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>(); 
		//GeneralDataset<String, String> predicted = new Dataset<String, String>();
		//EventFeatureFactory ff = new EventFeatureFactory();
		for(Example ex:testData) {
			LogInfo.begin_track("Example %s",ex.id);

			for(CoreMap sentence:ex.gold.get(SentencesAnnotation.class)) {
				
				//List<BioDatum> predictionInSentence = new ArrayList<BioDatum>();
				Set<Tree> entityNodes = null;
				if(prediction == null)
					entityNodes = Utils.getEntityNodesFromSentence(sentence);
				else
					entityNodes = Utils.getEntityNodesForSentenceFromDatum(prediction, sentence);
				
				LinearClassifier<String, String> classifier = new LinearClassifier<>(parameters.weights, parameters.featureIndex, parameters.labelIndex);
				List<BioDatum> dataset = ff.setFeaturesTest(sentence, entityNodes);

				for(BioDatum d:dataset) {
					Datum<String, String> newDatum = new BasicDatum<String, String>(d.getFeatures(),d.label());
					d.setPredictedLabel(classifier.classOf(newDatum));
				}
				predicted.addAll(dataset);
				if(printDebugInformation) {
					LogInfo.logs(sentence);
					LogInfo.logs(sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennString());
					LogInfo.logs(sentence.get(CollapsedCCProcessedDependenciesAnnotation.class));
				}

				if(printDebugInformation) {
					LogInfo.logs("\n---------GOLD EVENTS-------------------------");
					for(BioDatum d:dataset) 
						if(d.label().equals("E"))
							LogInfo.logs(d.features.getFeatureString() );
					
					LogInfo.logs("---------PREDICTIONS-------------------------");
					for(BioDatum d:dataset)
						if(d.predictedLabel().equals("E") || d.label().equals("E"))
							LogInfo.logs(String.format("Gold:  %s Predicted: %s, %-30s", d.label(), d.predictedLabel(), d.features.getFeatureString()));
					LogInfo.logs("------------------------------------------\n");
				}
			}
			
			LogInfo.end_track();
		}
		return predicted;
	}
	/*
    public List<BioDatum> Infer(List<Example> testData, Params parameters, FeatureExtractor ff) {
		List<BioDatum> predicted = new ArrayList<BioDatum>(); 
		//EventFeatureFactory ff = new EventFeatureFactory();
		for(Example ex:testData) {
			LogInfo.begin_track("Example %s",ex.id);

			for(CoreMap sentence:ex.gold.get(SentencesAnnotation.class)) {
				Set<Tree> entityNodes = null;
				if(prediction == null)
					entityNodes = Utils.getEntityNodesFromSentence(sentence);
				else
					entityNodes = Utils.getEntityNodesForSentenceFromDatum(prediction, sentence);
				
				List<BioDatum> test = ff.setFeaturesTest(sentence, entityNodes);
				if(printDebugInformation) {
					LogInfo.logs(sentence);
					LogInfo.logs(sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennString());
					LogInfo.logs(sentence.get(CollapsedCCProcessedDependenciesAnnotation.class));
				}

				List<BioDatum> testDataWithLabel = new ArrayList<BioDatum>();

				for (int i = 0; i < test.size(); i += parameters.getLabelIndex().size()) {
					testDataWithLabel.add(test.get(i));
				}
				//MaxEntModel maxEnt = new MaxEntModel(parameters.getLabelIndex(), parameters.getFeatureIndex(), parameters.getWeights());
				//maxEnt.decodeForEntity(testDataWithLabel, test);
				
				predicted.addAll(testDataWithLabel);
				
				if(printDebugInformation) {
					LogInfo.logs("\n---------GOLD EVENTS-------------------------");
					for(BioDatum d:testDataWithLabel) 
						if(d.label.equals("E"))
							LogInfo.logs(d.eventNode + ":" + d.label);
					
					LogInfo.logs("---------PREDICTIONS-------------------------");
					for(BioDatum d:testDataWithLabel)
						if(d.guessLabel.equals("E") || d.label.equals("E"))
							LogInfo.logs(String.format("%-30s [%s], Gold:  %s Predicted: %s", d.eventNode, d.eventNode.getSpan(), d.label, d.guessLabel));
					LogInfo.logs("------------------------------------------\n");
				}
			}
			LogInfo.end_track();
		}
		return predicted;
	}*/
}
