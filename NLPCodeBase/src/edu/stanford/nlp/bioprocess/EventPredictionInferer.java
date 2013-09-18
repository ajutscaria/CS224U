package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

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
	boolean printDebugInformation = false;
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
	
	public List<BioDatum> Infer(List<Example> testData, Params parameters, FeatureExtractor ff) {
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
}
