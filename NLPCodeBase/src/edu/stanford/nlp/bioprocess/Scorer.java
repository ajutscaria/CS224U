package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.Pair;

public class Scorer {
	boolean useF1 = true;
	/*
	public static double scoreEntityPredictionOld(Annotation gold, Annotation prediction) {
		int predicted = 0, predictedCorrect = 0, totalCorrect = 0;
		for(EntityMention entityGold: gold.get(EntityMentionsAnnotation.class)) {
			for(EntityMention entityPredicted: prediction.get(EntityMentionsAnnotation.class)) {
				if(entityGold.equals(entityPredicted)) {
					predictedCorrect++;
					break;
				}
			}
			totalCorrect++;
		}
		predicted = prediction.get(EntityMentionsAnnotation.class).size();

		double precision = predicted == 0 ? 0 : (double)predictedCorrect / predicted, 
				recall = totalCorrect == 0 ? 0 : (double)predictedCorrect / totalCorrect;
		return (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);
	}
	
	public static double scoreEntityPredictionOld(List<Example> examples) {
		double f1 = 0;
		for(Example ex:examples) {
			double score = scoreEntityPredictionOld(ex.gold, ex.prediction);
			if(score == 0)
				System.out.println(ex.id);
			f1 += score;
		}
		System.out.println("F1 score: " + f1 / examples.size());
		return f1 / examples.size();
	}
	*/

	public static double scoreEntityPrediction(Annotation gold, Annotation prediction) {
		int predictedCorrect = 0;
		for(EntityMention entityGold: gold.get(EntityMentionsAnnotation.class)) {
			for(EntityMention entityPredicted: prediction.get(EntityMentionsAnnotation.class)) {
				if(entityGold.equals(entityPredicted)) {
					predictedCorrect++;
					break;
				}
			}
		}
		return predictedCorrect;
	}
	
	public static double scoreEntityPrediction(List<Example> examples) {
		int predictedRight = 0;
		int totalCorrect = 0;
		int predicted = 0;
		for(Example ex:examples) {
			predictedRight += scoreEntityPrediction(ex.gold,ex.prediction);
			totalCorrect += ex.gold.get(EntityMentionsAnnotation.class).size();
			predicted += ex.prediction.get(EntityMentionsAnnotation.class).size();
		}
		double precision = predicted == 0 ? 0 : (double)predictedRight / predicted, 
				recall = totalCorrect == 0 ? 0 : (double)predictedRight / totalCorrect;
		double f1 = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);
		System.out.println(String.format("Precision : %f, Recall : %f, F1 score: %f", precision, recall, f1));
		return f1;
	}
	
	public static Pair<Double, Double> scoreEventPredictionOld(Annotation gold, Annotation prediction) {
		int predicted = 0, predictedCorrect = 0, totalCorrect = 0;
		List<EventMention> eventPredictedList = new ArrayList<EventMention>();
		for (EventMention eventPredicted: prediction.get(EventMentionsAnnotation.class)){
			eventPredictedList.add(eventPredicted);
		}
		for(EventMention eventGold: gold.get(EventMentionsAnnotation.class)) {
			boolean flag = false;
			for(EventMention eventPredicted: prediction.get(EventMentionsAnnotation.class)) {
				if(eventGold.equals(eventPredicted)) {
					predictedCorrect++;
					flag = true;
					//System.out.println("predicted right " + eventGold.prettyPrint());
					eventPredictedList.remove(eventGold);
					break;
				}
			}
			if (flag == false)
				System.out.println("not predicted " + eventGold.prettyPrint());
			totalCorrect++;
		}
		//for (EventMention eventNotPredicted: eventPredictedList){
	//		System.out.println("predicted wrong " + eventNotPredicted.prettyPrint());
	//	}
		predicted = prediction.get(EventMentionsAnnotation.class).size();

		double precision = predicted == 0 ? 0 : (double)predictedCorrect / predicted, 
				recall = totalCorrect == 0 ? 0 : (double)predictedCorrect / totalCorrect;
		Pair<Double, Double> precRecall = new Pair<Double, Double>(precision, recall);
		
		return precRecall;
	}
	
	public static double scoreEventPredictionOld(List<Example> examples) {
		double f1 = 0;
		double prec = 0;
		double recall = 0;
		System.out.println("finding scores");
		for(Example ex:examples) {
			System.out.println(ex.id);
			Pair<Double,Double> precRecallTemp = scoreEventPredictionOld(ex.gold,ex.prediction);
			prec += precRecallTemp.first;
			recall += precRecallTemp.second;
			double score = ((precRecallTemp.first + precRecallTemp.second) == 0) ? 0 : (2 * precRecallTemp.first * precRecallTemp.second / (precRecallTemp.first + precRecallTemp.second));
			if(score == 0)
				System.out.println(ex.id);
			f1 += score;
		}
		System.out.println("F1 score: " + f1 / examples.size());
		System.out.println("precision score: " + prec/examples.size());
		System.out.println("recall score: " + recall/examples.size());
		return f1 / examples.size();
	}
	
	public static Integer scoreEventPrediction(Annotation gold, Annotation prediction) {
		int predictedCorrect = 0;
		List<EventMention> eventPredictedList = new ArrayList<EventMention>();
		for (EventMention eventPredicted: prediction.get(EventMentionsAnnotation.class)){
			eventPredictedList.add(eventPredicted);
		}
		for(EventMention eventGold: gold.get(EventMentionsAnnotation.class)) {
			for(EventMention eventPredicted: prediction.get(EventMentionsAnnotation.class)) {
				if(eventGold.equals(eventPredicted)) {
					predictedCorrect++;
					//System.out.println("predicted right " + eventGold.prettyPrint());
					eventPredictedList.remove(eventPredicted);
					break;
				}
			}
			//if (flag == false)
				//System.out.println("not predicted " + eventGold.prettyPrint());
		}
		//for (EventMention eventNotPredicted: eventPredictedList){
		//	System.out.println("predicted wrong " + eventNotPredicted.prettyPrint());
		//}
		return predictedCorrect;
	}
	
	public static double scoreEventPrediction(List<Example> examples) {
		double f1 = 0;
		//double prec = 0;
		//double recall = 0;
		int predictedRight = 0;
		int totalCorrect = 0;
		int predicted = 0;
		System.out.println("finding scores");
		for(Example ex:examples) {
			System.out.println(ex.id);
			predictedRight += scoreEventPrediction(ex.gold,ex.prediction);
			totalCorrect += ex.gold.get(EventMentionsAnnotation.class).size();
			predicted += ex.prediction.get(EventMentionsAnnotation.class).size();
		}
		System.out.println("predicted right" + predictedRight);
		System.out.println("predicted "+ predicted);
		System.out.println("total right "+ totalCorrect);
		double precision = predicted == 0 ? 0 : (double)predictedRight / predicted, 
				recall = totalCorrect == 0 ? 0 : (double)predictedRight / totalCorrect;
		f1 = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);
	
		System.out.println("F1 score: " + f1 );
		System.out.println("precision score: " + precision);
		System.out.println("recall score: " + recall);
		return f1 / examples.size();
	}
}
