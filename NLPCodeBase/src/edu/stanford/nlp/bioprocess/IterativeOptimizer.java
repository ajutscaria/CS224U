package edu.stanford.nlp.bioprocess;

import java.util.List;

import edu.stanford.nlp.util.Triple;

public class IterativeOptimizer {
	public Triple<Double, Double, Double> optimize(List<Example> train, List<Example> test) {
		Learner eventLearner = new EventPredictionLearner();
		FeatureExtractor eventFeatureFactory = new EventFeatureFactory();
		Inferer inferer = new EventPredictionInferer();
		Params param = eventLearner.learn(train, eventFeatureFactory);
		List<Datum> predicted = inferer.Infer(test, param, eventFeatureFactory);
		Triple<Double, Double, Double> triple = Scorer.score(predicted);
		
		Learner entityLearner = new EntityPredictionLearner();
		FeatureExtractor entityFeatureFactory = new EntityFeatureFactory();
		Inferer entityInferer = new EntityPredictionInferer(predicted);
		Params entityParams = entityLearner.learn(train, entityFeatureFactory);
		List<Datum> predictedEntities = entityInferer.Infer(test, entityParams, entityFeatureFactory);
		Triple<Double, Double, Double> entityTriple = Scorer.score(test, predictedEntities);

		return entityTriple;
	}
}
