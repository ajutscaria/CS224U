package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;

public class EventPredictionInference {
	public void BaselineInfer(List<Example> examples) {
		for(Example example:examples) {
			for(CoreMap sentence: example.gold.get(SentencesAnnotation.class)) {
				for(CoreLabel token: sentence.get(TokensAnnotation.class)) {
					if(token.get(PartOfSpeechAnnotation.class).startsWith("VB")) {
						EventMention event = new EventMention("obj", sentence, new Span(token.index()-1, token.index()));
						Utils.addAnnotation(example.prediction, event);
					}
				}
			}
		}
	}
	
    public double Infer(List<Example> testData, Params parameters) {
		List<Datum> predicted = new ArrayList<Datum>(); 
		FeatureFactoryEvents ff = new FeatureFactoryEvents();
		for(Example ex:testData) {
			System.out.println(String.format("==================EXAMPLE %s======================",ex.id));
			for(CoreMap sentence:ex.gold.get(SentencesAnnotation.class)) {
				List<Datum> test = ff.setFeaturesTest(sentence);
				System.out.println(sentence);
				sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
				System.out.println(sentence.get(CollapsedCCProcessedDependenciesAnnotation.class));

				List<Datum> testDataWithLabel = new ArrayList<Datum>();

				for (int i = 0; i < test.size(); i += parameters.getLabelIndex().size()) {
					testDataWithLabel.add(test.get(i));
				}
				MaxEntModel maxEnt = new MaxEntModel(parameters.getLabelIndex(), parameters.getFeatureIndex(), parameters.getWeights());
				maxEnt.decodeForEntity(testDataWithLabel, test);
				
				predicted.addAll(testDataWithLabel);
				

				
				System.out.println("\n---------GOLD ENTITIES-------------------------");
				for(Datum d:testDataWithLabel) 
					if(d.label.equals("E"))
						System.out.println(d.eventNode + ":" + d.label);
				
				System.out.println("---------PREDICTIONS-------------------------");
				for(Datum d:testDataWithLabel)
					if(d.guessLabel.equals("E") || d.label.equals("E"))
						System.out.println(String.format("%-30s [%s], Gold:  %s Predicted: %s", d.word, d.entityNode.getSpan(), d.label, d.guessLabel));
				System.out.println("------------------------------------------\n");
			}
		}
		
				
		double f1 = Scorer.score(predicted);
		
		return f1;
	}
}
