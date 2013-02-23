package edu.stanford.nlp.bioprocess;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.pipeline.Annotation;

public class Scorer {
	boolean useF1 = true;
	
	public static double scoreEntityPrediction(Annotation gold, Annotation prediction) {
		int predicted = 0, predictedCorrect = 0, totalCorrect = 0;
		for(EntityMention entityGold: gold.get(EntityMentionsAnnotation.class)) {
			for(EntityMention entityPredicted: prediction.get(EntityMentionsAnnotation.class)) {
				if(entityGold.equals(entityPredicted)) {
					predictedCorrect++;
				}
			}
			totalCorrect++;
		}
		predicted = prediction.get(EntityMentionsAnnotation.class).size();
		double precision = predictedCorrect / predicted, recall = predictedCorrect / totalCorrect;
		return 2 * precision * recall / (precision + recall);
	}
}
