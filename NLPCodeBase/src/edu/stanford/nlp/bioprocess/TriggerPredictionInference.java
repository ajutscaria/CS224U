package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.util.CoreMap;

public class TriggerPredictionInference {
	public void baselineInfer(List<Example> examples) {
		//CrossValidationSplit CV = new CrossValidationSplit(examples,10);
		//List<Example> test = CV.GetTestExamples(1);
		//for (Example t:test){
		//	System.out.println(t.id);
		//}
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
}
