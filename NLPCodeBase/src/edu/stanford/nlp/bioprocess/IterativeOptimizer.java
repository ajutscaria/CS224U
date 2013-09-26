package edu.stanford.nlp.bioprocess;

import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import fig.basic.LogInfo;

public class IterativeOptimizer {
	public static final String ANSI_RESET = "\u001B[0m";
	public static final String ANSI_BLACK = "\u001B[30m";
	public static final String ANSI_RED = "\u001B[31m";
	public static final String ANSI_GREEN = "\u001B[32m";
	public static final String ANSI_YELLOW = "\u001B[33m";
	public static final String ANSI_BLUE = "\u001B[34m";
	public static final String ANSI_PURPLE = "\u001B[35m";
	public static final String ANSI_CYAN = "\u001B[36m";
	public static final String ANSI_WHITE = "\u001B[37m";

	public Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> optimize(List<Example> train, List<Example> test, boolean useLexicalFeatures) {
		LogInfo.begin_track("Basiccc trigger prediction");
		Learner eventLearner = new Learner();
		FeatureExtractor eventFeatureFactory = new EventFeatureFactory(useLexicalFeatures);
		Inferer inferer = new EventPredictionInferer();
		Params param = eventLearner.learn(train, eventFeatureFactory);
		List<BioDatum> predicted = inferer.Infer(test, param, eventFeatureFactory);
		Triple<Double, Double, Double> triple = Scorer.score(predicted);

		LogInfo.logs("Score: Basic trigger prediction - " + triple);
		LogInfo.end_track();

		Learner entityLearner = new Learner();
		FeatureExtractor entityFeatureFactory = new EntityFeatureFactory(useLexicalFeatures);

		Inferer entityInferer = new EntityStandaloneInferer();
		FeatureExtractor entityStandaloneFeatureFactory = new EntityStandaloneFeatureFactory(useLexicalFeatures);
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
			eventFeatureFactory = new EventExtendedFeatureFactory(useLexicalFeatures);
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

	public Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> runPipelinePrediction(List<Example> train, List<Example> test, boolean useLexicalFeatures, String model) {
		FeatureExtractor eventFeatureFactory = new EventFeatureFactory(useLexicalFeatures);
		Inferer inferer = new EventPredictionInferer();
		Params param = (Params) Utils.readObject(Main.EVENT_STANDALONE_MODEL);
		List<BioDatum> predicted = inferer.Infer(test, param, eventFeatureFactory);
		Triple<Double, Double, Double> triple = Scorer.score(predicted);

		FeatureExtractor entityFeatureFactory = new EntityFeatureFactory(useLexicalFeatures);

		Inferer entityInferer = new EntityStandaloneInferer();
		FeatureExtractor entityStandaloneFeatureFactory = new EntityStandaloneFeatureFactory(useLexicalFeatures);
		Params entityStandaloneParams = (Params) Utils.readObject(Main.ENTITY_STANDALONE_MODEL);
		List<BioDatum> predictedStandaloneEntities = entityInferer.Infer(test, entityStandaloneParams, entityStandaloneFeatureFactory);

		Triple<Double, Double, Double> entityTriple = null;
		for(int i = 0; i < 1; i++) {
			LogInfo.begin_track("Entity prediction");
			entityInferer = new EntityPredictionInferer(predicted);
			Params entityParams = (Params) Utils.readObject(Main.ENTITY_MODEL);
			List<BioDatum> predictedEntities = entityInferer.Infer(test, entityParams, entityFeatureFactory);
			entityTriple = Scorer.scoreEntities(test, predictedEntities);

			LogInfo.logs("Score: Entity prediction - " + entityTriple);
			LogInfo.end_track();

			predictedEntities.addAll(predictedStandaloneEntities);

			LogInfo.begin_track("Extended trigger prediction");
			inferer = new EventPredictionInferer(predictedEntities);
			eventFeatureFactory = new EventExtendedFeatureFactory(useLexicalFeatures);
			param = (Params) Utils.readObject(Main.EVENT_MODEL);
			predicted = inferer.Infer(test, param, eventFeatureFactory);
			triple = Scorer.score(predicted);

			LogInfo.logs("Score: Extended trigger prediction - " + triple);
			LogInfo.end_track();
			//break;
		}

		entityInferer = new EntityPredictionInferer(predicted);
		Params entityParams = (Params) Utils.readObject(Main.ENTITY_MODEL);
		List<BioDatum> predictedEntities = entityInferer.Infer(test, entityParams, entityFeatureFactory);
		entityTriple = Scorer.scoreEntities(test, predictedEntities);

		LogInfo.logs("Entity prediction - " + entityTriple);

		//Learner eventRelationLearner = new Learner();
		EventRelationFeatureFactory eventRelationFeatureFactory = new EventRelationFeatureFactory(useLexicalFeatures, model);
		EventRelationInferer relationInferer = new EventRelationInferer("global");

		Params eventParam = (Params) Utils.readObject(Main.EVENT_RELATION_GLOBAL_MODEL);
		List<BioDatum> result = new ArrayList<BioDatum>();

		for(Example ex:test) {
			LogInfo.logs("Example" + ex.id);
			LogInfo.logs("Gold Events:");
			for(EventMention m:ex.gold.get(EventMentionsAnnotation.class)) {
				LogInfo.logs("\t" + m.getTreeNode());
			}
			List<EventMention> eventsInExample = new ArrayList<EventMention>();
			LogInfo.logs("Predicted Events:");
			for(BioDatum d:predicted) {
				if(d.getExampleID().equals(ex.id) && d.guessLabel.equals("E")) {
					IntPair span = d.eventNode.getSpan();
					Span evtSpan = new Span(span.getSource(), span.getTarget() + 1);
					LogInfo.logs("\t" + d.eventNode);
					ArgumentMention mention = new EventMention(Utils.getText(d.eventNode) + "_" + eventsInExample.size(), d.sentence, evtSpan);
					IndexedWord head = Utils.findDependencyNode(d.sentence, d.eventNode);

					mention.setHeadInDependencyTree(head);
					mention.setTreeNode(d.eventNode);
					eventsInExample.add((EventMention)mention);
				}
			}

			result.addAll(relationInferer.PipelineInfer(ex, eventsInExample, eventParam, eventRelationFeatureFactory, model,
					true, true, true, true, 0.0,0.5,0,0,1.0,0.0, 0.5));
		}
		Triple<Double, Double, Double> resultTriple;

		resultTriple = Scorer.scoreEventRelationsPipeline(test, result);

		return new Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>>(resultTriple, new Triple<Double, Double, Double>(0.0,0.0,0.0));
	}

	public void runInference(List<Example> examples) {
		boolean useLexicalFeatures = true;
		String model = "global";
		FeatureExtractor eventFeatureFactory = new EventFeatureFactory(useLexicalFeatures);
		Inferer inferer = new EventPredictionInferer();
		Params param = (Params) Utils.readObject(Main.EVENT_STANDALONE_MODEL);
		List<BioDatum> predicted = inferer.Infer(examples, param, eventFeatureFactory);
		//Triple<Double, Double, Double> triple = Scorer.score(predicted);

		FeatureExtractor entityFeatureFactory = new EntityFeatureFactory(useLexicalFeatures);

		Inferer entityInferer = new EntityStandaloneInferer();
		FeatureExtractor entityStandaloneFeatureFactory = new EntityStandaloneFeatureFactory(useLexicalFeatures);
		Params entityStandaloneParams = (Params) Utils.readObject(Main.ENTITY_STANDALONE_MODEL);
		List<BioDatum> predictedStandaloneEntities = entityInferer.Infer(examples, entityStandaloneParams, entityStandaloneFeatureFactory);

		Triple<Double, Double, Double> entityTriple = null;
		for(int i = 0; i < 1; i++) {
			//LogInfo.begin_track("Entity prediction");
			entityInferer = new EntityPredictionInferer(predicted);
			Params entityParams = (Params) Utils.readObject(Main.ENTITY_MODEL);
			List<BioDatum> predictedEntities = entityInferer.Infer(examples, entityParams, entityFeatureFactory);
			entityTriple = Scorer.scoreEntities(examples, predictedEntities);

			//LogInfo.logs("Score: Entity prediction - " + entityTriple);
			//LogInfo.end_track();

			predictedEntities.addAll(predictedStandaloneEntities);

			//LogInfo.begin_track("Extended trigger prediction");
			inferer = new EventPredictionInferer(predictedEntities);
			eventFeatureFactory = new EventExtendedFeatureFactory(useLexicalFeatures);
			param = (Params) Utils.readObject(Main.EVENT_MODEL);
			predicted = inferer.Infer(examples, param, eventFeatureFactory);
			//triple = Scorer.score(predicted);

			//LogInfo.logs("Score: Extended trigger prediction - " + triple);
			//LogInfo.end_track();
			//break;
		}

		entityInferer = new EntityPredictionInferer(predicted);
		Params entityParams = (Params) Utils.readObject(Main.ENTITY_MODEL);
		List<BioDatum> predictedEntities = entityInferer.Infer(examples, entityParams, entityFeatureFactory);
		entityTriple = Scorer.scoreEntities(examples, predictedEntities);

		//LogInfo.logs("Entity prediction - " + entityTriple);

		//Learner eventRelationLearner = new Learner();
		EventRelationFeatureFactory eventRelationFeatureFactory = new EventRelationFeatureFactory(useLexicalFeatures, model);
		EventRelationInferer relationInferer = new EventRelationInferer("global");

		Params eventParam = (Params) Utils.readObject(Main.EVENT_RELATION_GLOBAL_MODEL);
		List<BioDatum> result = new ArrayList<BioDatum>();

		for(Example ex:examples) {
			List<EventMention> eventsInExample = new ArrayList<EventMention>();
			for(BioDatum d:predicted) {
				if(d.getExampleID().equals(ex.id) && d.guessLabel.equals("E")) {
					IntPair span = d.eventNode.getSpan();
					Span evtSpan = new Span(span.getSource(), span.getTarget() + 1);

					ArgumentMention mention = new EventMention(Utils.getText(d.eventNode) + "_" + eventsInExample.size(), d.sentence, evtSpan);
					IndexedWord head = Utils.findDependencyNode(d.sentence, d.eventNode);

					mention.setHeadInDependencyTree(head);
					mention.setTreeNode(d.eventNode);
					eventsInExample.add((EventMention)mention);
				}
			}

			if(eventsInExample.size() >= 1) {
				result.addAll(relationInferer.PipelineInfer(ex, eventsInExample, eventParam, eventRelationFeatureFactory, model,
						true, true, true, true, 0.0,0.5,0,0,1.0,0.0, 0.5));

				System.out.println("\n   Predicted " + ANSI_GREEN + "events" +
						ANSI_RESET + " and associated " + ANSI_BLUE + "entities" +
						ANSI_RESET + ":");
				for(BioDatum datumEvent:predicted) {
					if(datumEvent.guessLabel.equals("E")) {
						if(datumEvent.getExampleID().equals(ex.id) && datumEvent.guessLabel.equals("E")) {
							System.out.println(ANSI_GREEN + "\t" + Utils.getText(datumEvent.eventNode));
						}

						for(BioDatum datumEntity:predictedEntities) {
							if(datumEntity.guessLabel.equals("E") &&
									datumEntity.eventNode == datumEvent.eventNode) {
								System.out.println(ANSI_BLUE + "\t\t" + Utils.getText(datumEntity.entityNode));
							}
						}
					}
				}
				if(eventsInExample.size() > 1) {
					LogInfo.logs(ANSI_RESET + "\n   Predicted event relations:");
					for(BioDatum eventRelation:result) {
						if(!eventRelation.guessLabel.equals("NONE")) {
							System.out.println(ANSI_CYAN + "\t" + Utils.getText(eventRelation.event1.getTreeNode()) + " - " +
									Utils.getText(eventRelation.event2.getTreeNode()) + " : " + eventRelation.guessLabel);
						}
					}
				}
			}
			else {
				System.out.println("\n   " + ANSI_GREEN + "No event triggers predicted.");
			}
			System.out.println(ANSI_RESET);
		}
	}
}
