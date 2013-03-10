package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;

public class EntityStandaloneInferer extends Inferer{
	private boolean printDebugInformation = true;
	
	@Override
	public List<Datum> BaselineInfer(List<Example> examples, Params parameters,
			FeatureExtractor ff) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<Datum> Infer(List<Example> testData, Params parameters, FeatureExtractor ff) {
		List<Datum> predicted = new ArrayList<Datum>();
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

				List<Datum> test = ff.setFeaturesTest(sentence, eventNodes);
				
				List<Datum> testDataEvent = new ArrayList<Datum>();
				for(Datum d:test)
					testDataEvent.add(d);

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
				
				LogInfo.logs(sentence);
					//sentence.get(TreeCoreAnnotations.TreeAnnotation.class).pennPrint();
					
				LogInfo.logs("\n---------GOLD ENTITIES-------------------------");
				for(Datum d:testDataWithLabel) 
					if(d.label.equals("E"))
						LogInfo.logs(d.entityNode + ":" + d.label);
				
				LogInfo.logs("---------PREDICTIONS-------------------------");
				for(Datum d:testDataWithLabel)
					if(d.guessLabel.equals("E") || d.label.equals("E"))
						LogInfo.logs(String.format("%-30s [%s], Gold:  %s Predicted: %s", d.word, d.entityNode.getSpan(), d.label, d.guessLabel));
				LogInfo.logs("------------------------------------------\n");
			}
			
			LogInfo.end_track();
		}
		return predicted;
	}
}
