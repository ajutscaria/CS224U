package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

public class EntityPredictionInferer extends Inferer {
	public List<Datum> BaselineInfer(List<Example> examples, Params parameters) {
		List<Datum> predicted = new ArrayList<Datum>();
		EntityFeatureFactory ff = new EntityFeatureFactory();
		for(Example example:examples) {
			System.out.println(String.format("==================EXAMPLE %s======================", example.id));
			for(CoreMap sentence: example.gold.get(SentencesAnnotation.class)) {
				List<Datum> test = ff.setFeaturesTest(sentence);
				for(EventMention event:sentence.get(EventMentionsAnnotation.class)) {
					System.out.println("------------------Event " + event.getValue()+"--------------");
					List<Datum> testDataEvent = new ArrayList<Datum>();
					for(Datum d:test)
						if(d.eventNode == event.getTreeNode()) {
							testDataEvent.add(d);
						}
					List<Datum> testDataWithLabel = new ArrayList<Datum>();
	
					for (int i = 0; i < testDataEvent.size(); i += parameters.getLabelIndex().size()) {
						testDataWithLabel.add(testDataEvent.get(i));
					}
					
					for(Datum d:testDataWithLabel) {
						if(d.entityNode.value().equals("NP") && Utils.isNodesRelated(sentence, d.entityNode, event))
							d.guessLabel = "E";
						else
							d.guessLabel = "O";
					}
					predicted.addAll(testDataWithLabel);
					
					System.out.println(sentence);
					//sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
					
					System.out.println("\n---------GOLD ENTITIES-------------------------");
					for(Datum d:testDataWithLabel) 
						if(d.label.equals("E"))
							System.out.println(d.entityNode + ":" + d.label);
					
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
	
	public List<Datum> Infer(List<Example> testData, Params parameters) {
		List<Datum> predicted = new ArrayList<Datum>();
		EntityFeatureFactory ff = new EntityFeatureFactory();
		for(Example ex:testData) {
			System.out.println(String.format("==================EXAMPLE %s======================",ex.id));
			//IdentityHashSet<Tree> entities = Utils.getEntityNodes(ex);
			for(CoreMap sentence:ex.gold.get(SentencesAnnotation.class)) {
				List<Datum> test = ff.setFeaturesTest(sentence);
				
				for(EventMention event:sentence.get(EventMentionsAnnotation.class)) {
					System.out.println("------------------Event " + event.getValue()+"--------------");
					List<Datum> testDataEvent = new ArrayList<Datum>();
					for(Datum d:test)
						if(d.eventNode == event.getTreeNode()) {
							//System.out.println(d.entityNode);
							testDataEvent.add(d);
						}
					List<Datum> testDataWithLabel = new ArrayList<Datum>();
	
					for (int i = 0; i < testDataEvent.size(); i += parameters.getLabelIndex().size()) {
						testDataWithLabel.add(testDataEvent.get(i));
					}
					MaxEntModel viterbi = new MaxEntModel(parameters.getLabelIndex(), parameters.getFeatureIndex(), parameters.getWeights());
					viterbi.decodeForEntity(testDataWithLabel, testDataEvent);
					
					IdentityHashMap<Tree, Pair<Double, String>> map = new IdentityHashMap<Tree, Pair<Double, String>>();
	
					for(Datum d:testDataWithLabel) {
						if (Utils.subsumesEvent(d.entityNode, sentence)) {
							map.put(d.entityNode, new Pair<Double, String>(0.0, "O"));
						} else {
							map.put(d.entityNode, new Pair<Double, String>(d.getProbability(), d.guessLabel));
						}
					}
					
					DynamicProgramming dynamicProgrammer = new DynamicProgramming(sentence, map, testDataWithLabel);
					dynamicProgrammer.calculateLabels();
					
					predicted.addAll(testDataWithLabel);
					
					System.out.println(sentence);
					//sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
					
					System.out.println("\n---------GOLD ENTITIES-------------------------");
					for(Datum d:testDataWithLabel) 
						if(d.label.equals("E"))
							System.out.println(d.entityNode + ":" + d.label);
					
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
