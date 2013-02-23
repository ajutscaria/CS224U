package edu.stanford.nlp.bioprocess;

import java.util.List;

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

		double precision = predicted == 0 ? 0 : (double)predictedCorrect / predicted, 
				recall = totalCorrect == 0 ? 0 : (double)predictedCorrect / totalCorrect;
		return (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);
	}
	
	public static double scoreEntityPrediction(List<Example> examples) {
		double f1 = 0;
		for(Example ex:examples) {
			double score = scoreEntityPrediction(ex.gold, ex.prediction);
			if(score == 0)
				System.out.println(ex.id);
			f1 += score;
		}
		return f1 / examples.size();
	}
}
