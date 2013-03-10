package edu.stanford.nlp.bioprocess;

import java.util.List;

import edu.stanford.nlp.util.Triple;
import fig.basic.LogInfo;

public class IterativeOptimizer {
	public Triple<Double, Double, Double> optimize(List<Example> train, List<Example> test) {
		Learner eventLearner = new EventPredictionLearner();
		FeatureExtractor eventFeatureFactory = new EventFeatureFactory();
		Inferer inferer = new EventPredictionInferer();
		Params param = eventLearner.learn(train, eventFeatureFactory);
		List<Datum> predicted = inferer.Infer(test, param, eventFeatureFactory);
		Triple<Double, Double, Double> triple = Scorer.score(predicted);
		
		LogInfo.logs("Basic trigger prediction - " + triple);
		
		Learner entityLearner = new EntityPredictionLearner();
		FeatureExtractor entityFeatureFactory = new EntityFeatureFactory();
		
		Triple<Double, Double, Double> entityTriple = null;
		for(int i = 0; i <3; i++) {
			Inferer entityInferer = new EntityPredictionInferer(predicted);
			Params entityParams = entityLearner.learn(train, entityFeatureFactory);
			List<Datum> predictedEntities = entityInferer.Infer(test, entityParams, entityFeatureFactory);
			entityTriple = Scorer.scoreEntities(test, predictedEntities);
			
			LogInfo.logs("Entity prediction - " + entityTriple);
			
			inferer = new EventPredictionInferer(predictedEntities);
			eventFeatureFactory = new EventExtendedFeatureFactory();
			param = eventLearner.learn(train, eventFeatureFactory);
			predicted = inferer.Infer(test, param, eventFeatureFactory);
			triple = Scorer.scoreEvents(test, predicted);
			
			LogInfo.logs("Trigger prediction - " + triple);
			//break;
		}
		
		return entityTriple;
	}
	
	public Triple<Double, Double, Double> predictEntity(List<Example> train, List<Example> test) {
		
		Learner entityLearner = new EntityPredictionLearner();
		FeatureExtractor entityFeatureFactory = new EntityStandaloneFeatureFactory();
		
		Triple<Double, Double, Double> entityTriple = null;
		for(int i = 0; i <3; i++) {
			Inferer entityInferer = new EntityStandaloneInferer();
			Params entityParams = entityLearner.learn(train, entityFeatureFactory);
			List<Datum> predictedEntities = entityInferer.Infer(test, entityParams, entityFeatureFactory);
			entityTriple = Scorer.scoreStandaloneEntities(test, predictedEntities);
			
			LogInfo.logs("Entity prediction - " + entityTriple);
			//break;
		}
		
		return entityTriple;
	}
}
