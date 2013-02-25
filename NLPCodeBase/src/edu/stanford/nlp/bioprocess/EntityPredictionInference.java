package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;

public class EntityPredictionInference {
	public void baselineInfer(List<Example> examples) {
		for(Example example:examples) {
			for(CoreMap sentence: example.gold.get(SentencesAnnotation.class)) {
				for(CoreLabel token: sentence.get(TokensAnnotation.class)) {
					if(token.get(PartOfSpeechAnnotation.class).startsWith("NN")) {
						EntityMention entity = new EntityMention("obj", sentence, new Span(token.index()-1, token.index()));
						entity.setHeadTokenSpan(entity.getExtent());
						Utils.addAnnotation(example.prediction, entity);
					}
				}
			}
		}
	}
	
	public double MEMMInfer(List<Example> examples, double[][] weights) {
		FeatureFactory ff = new FeatureFactory();
		List<Datum> test = ff.setFeaturesTest(examples);
	    LogConditionalObjectiveFunction obj = new LogConditionalObjectiveFunction(test);
	    //System.out.println(obj.labelIndex.size());
		List<Datum> testData = new ArrayList<Datum>();
		testData.add(test.get(0));
		for (int i = 1; i < test.size(); i += obj.labelIndex.size()) {
			testData.add(test.get(i));
		}
		Viterbi viterbi = new Viterbi(obj.labelIndex, obj.featureIndex, weights);
		viterbi.decode(testData, test);
		
		for(Datum d:testData)
			System.out.println(String.format("%-20s Gold: %s, Predicted: %s", d.word, d.label, d.guessLabel));
		
		double f1 = Scorer.score(testData);
		
		return f1;//testData;
	}
}
