package edu.stanford.nlp.bioprocess;

import java.util.List;

import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import fig.basic.LogInfo;

public class IterativeOptimizer {
	public Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> optimize(List<Example> train, List<Example> test) {
		LogInfo.begin_track("Basiccc trigger prediction");
		EventPredictionLearner eventLearner = new EventPredictionLearner();
		FeatureExtractor eventFeatureFactory = new EventFeatureFactory();
		Inferer inferer = new EventPredictionInferer();
		Params param = eventLearner.learn(train, eventFeatureFactory);
		List<BioDatum> predicted = inferer.Infer(test, param, eventFeatureFactory);
		Triple<Double, Double, Double> triple = Scorer.score(predicted);
		
		LogInfo.logs("Score: Basic trigger prediction - " + triple);
		LogInfo.end_track();
		
		Learner entityLearner = new EntityPredictionLearner();
		FeatureExtractor entityFeatureFactory = new EntityFeatureFactory();
		
		Inferer entityInferer = new EntityStandaloneInferer();
		FeatureExtractor entityStandaloneFeatureFactory = new EntityStandaloneFeatureFactory();
		Params entityStandaloneParams = entityLearner.learn(train, entityStandaloneFeatureFactory);
		List<BioDatum> predictedStandaloneEntities = entityInferer.Infer(test, entityStandaloneParams, entityStandaloneFeatureFactory);
		
		Triple<Double, Double, Double> entityTriple = null;
		for(int i = 0; i < 1; i++) {
			LogInfo.begin_track("Entity prediction");
			entityInferer = new EntityPredictionInferer(predicted);
			Params entityParams = entityLearner.learn(train, entityFeatureFactory);
			List<BioDatum> predictedEntities = entityInferer.Infer(test, entityParams, entityFeatureFactory);
			entityTriple = Scorer.scoreEntities(test, predictedEntities);
			
			LogInfo.logs("Score: Entity prediction - " + entityTriple);
			LogInfo.end_track();
			
			predictedEntities.addAll(predictedStandaloneEntities);
			
			LogInfo.begin_track("Extended trigger prediction");
			inferer = new EventPredictionInferer(predictedEntities);
			eventFeatureFactory = new EventExtendedFeatureFactory();
			param = eventLearner.learn(train, eventFeatureFactory);
			predicted = inferer.Infer(test, param, eventFeatureFactory);
			triple = Scorer.score(predicted);
			
			LogInfo.logs("Score: Extended trigger prediction - " + triple);
			LogInfo.end_track();
			//break;
		}
		
		entityInferer = new EntityPredictionInferer(predicted);
		Params entityParams = entityLearner.learn(train, entityFeatureFactory);
		List<BioDatum> predictedEntities = entityInferer.Infer(test, entityParams, entityFeatureFactory);
		entityTriple = Scorer.scoreEntities(test, predictedEntities);
		
		LogInfo.logs("Entity prediction - " + entityTriple);
		
		return new Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>>(triple, entityTriple);
	}
}
