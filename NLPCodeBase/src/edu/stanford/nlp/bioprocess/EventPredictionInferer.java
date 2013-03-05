package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;

public class EventPredictionInferer extends Inferer {
	boolean printDebugInformation = true;
	
	public List<Datum> BaselineInfer(List<Example> examples, Params parameters) {
		List<Datum> predicted = new ArrayList<Datum>(); 
		EventFeatureFactory ff = new EventFeatureFactory();
		for(Example example:examples) {
			for(CoreMap sentence: example.gold.get(SentencesAnnotation.class)) {
				List<Datum> test = ff.setFeaturesTest(sentence);
				List<Datum> testDataWithLabel = new ArrayList<Datum>();

				for (int i = 0; i < test.size(); i += parameters.getLabelIndex().size()) {
					testDataWithLabel.add(test.get(i));
				}
				
				for(Datum d:testDataWithLabel)
					if(d.entityNode.isPreTerminal() && d.entityNode.value().startsWith("VB"))
						d.guessLabel = "E";
					else
						d.guessLabel = "O";
				predicted.addAll(testDataWithLabel);
			}
		}
		return predicted;
	}
	
    public List<Datum> Infer(List<Example> testData, Params parameters) {
		List<Datum> predicted = new ArrayList<Datum>(); 
		EventFeatureFactory ff = new EventFeatureFactory();
		for(Example ex:testData) {
			if(printDebugInformation)
				System.out.println(String.format("==================EXAMPLE %s======================",ex.id));
			for(CoreMap sentence:ex.gold.get(SentencesAnnotation.class)) {
				List<Datum> test = ff.setFeaturesTest(sentence);
				if(printDebugInformation) {
					System.out.println(sentence);
					sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
					System.out.println(sentence.get(CollapsedCCProcessedDependenciesAnnotation.class));
				}

				List<Datum> testDataWithLabel = new ArrayList<Datum>();

				for (int i = 0; i < test.size(); i += parameters.getLabelIndex().size()) {
					testDataWithLabel.add(test.get(i));
				}
				MaxEntModel maxEnt = new MaxEntModel(parameters.getLabelIndex(), parameters.getFeatureIndex(), parameters.getWeights());
				maxEnt.decodeForEntity(testDataWithLabel, test);
				
				predicted.addAll(testDataWithLabel);
				
				if(printDebugInformation) {
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
		}
		return predicted;
	}
}
