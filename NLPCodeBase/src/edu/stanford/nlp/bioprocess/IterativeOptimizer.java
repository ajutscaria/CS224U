package edu.stanford.nlp.bioprocess;

import java.util.List;

import edu.stanford.nlp.util.Triple;

public class IterativeOptimizer {
	public void optimize(List<Example> train, List<Example> test) {
		Learner eventLearner = new EventPredictionLearner();
		FeatureExtractor eventFeatureFactory = new EventFeatureFactory();
		Inferer inferer = new EventPredictionInferer();
		Params param = eventLearner.learn(train, eventFeatureFactory);
		List<Datum> predicted = inferer.Infer(test, param, eventFeatureFactory);
		Triple<Double, Double, Double> triple = Scorer.score(predicted);
		
		
		System.out.println(triple);
	}
}
