package edu.stanford.nlp.bioprocess;

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
						EntityMention entity = new EntityMention("obj", sentence, new Span(token.index(), token.index() + 1));
						Utils.addAnnotation(example.prediction, entity);
					}
				}
			}
		}
	}
}
