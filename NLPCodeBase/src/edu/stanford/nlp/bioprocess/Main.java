package edu.stanford.nlp.bioprocess;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Triple;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;

public class Main implements Runnable {

	//public static class Options {
		@Option(gloss="Where to read the property file from") public String propertyFile;
		@Option(gloss="The running mode: event, entity, or em") public String mode;
	//}	
	//public static Options opts = new Options();

	public void runPrediction(HashMap<String, String> groups, FeatureExtractor featureFactory, Learner learner, Inferer inferer, Scorer scorer) {
		int NumCrossValidation = 10;
		double[] precisionTrain = new double[NumCrossValidation], recallTrain = new double[NumCrossValidation], f1Train = new double[NumCrossValidation], 
				precisionDev = new double[NumCrossValidation], recallDev = new double[NumCrossValidation], f1Dev = new double[NumCrossValidation],
				precisionBaseline = new double[NumCrossValidation], recallBaseline = new double[NumCrossValidation], f1Baseline = new double[NumCrossValidation];

		//Flags to indicate if evaluation of model should be run on training set, baseline and dev-test set.
		boolean evaluateTrain = false, evaluateBaseline = false, evaluateDev = true;
		//Flags to control sample on which test is to be run. useSmallSample runs on 2 sample files, while useOneLoop runs one fold of CV.
		//refreshDataFile is to re-generate the bpa (bio process annotation) file
		boolean useSmallSample = false, useOneLoop = false, refreshDataFile = true;
		//useSmallSample = true;
		//useOneLoop = true;
		//refreshDataFile = true;

		BioprocessDataset dataset = loadDataSet(groups, useSmallSample, refreshDataFile);
		
		if(useSmallSample) {
			Params param = learner.learn(dataset.examples("sample"), featureFactory);
			List<BioDatum> predicted = inferer.Infer(dataset.examples("sample"), param, featureFactory);
			Triple<Double, Double, Double> triple = Scorer.score(predicted);
			LogInfo.logs("Precision : " + triple.first);
			LogInfo.logs("Recall    : " + triple.second);
			LogInfo.logs("F1 score  : " + triple.third);
		}
		else {
			CrossValidationSplit split = new CrossValidationSplit(dataset.examples("train"), NumCrossValidation);
			LogInfo.begin_track("Cross validation");
			for(int i = 1; i <= NumCrossValidation; i++) {
				LogInfo.begin_track("Iteration " + i);
				LogInfo.begin_track("Train");
				Params param = learner.learn(split.GetTrainExamples(i), featureFactory);
				LogInfo.end_track();
				List<BioDatum> predicted;
				Triple<Double, Double, Double> triple;
				LogInfo.begin_track("Inference");
				if(evaluateTrain) {	
					LogInfo.begin_track("Train");
					predicted = inferer.Infer(split.GetTrainExamples(i), param, featureFactory);
					triple = Scorer.score(predicted);
					precisionTrain[i-1] = triple.first; recallTrain[i-1] = triple.second; f1Train[i-1] = triple.third;
					LogInfo.end_track();
				}
				if(evaluateBaseline) {
					LogInfo.begin_track("baseline");
					predicted = inferer.BaselineInfer(split.GetTestExamples(i), param, featureFactory);
					triple = Scorer.score(predicted);
					precisionBaseline[i-1] = triple.first; recallBaseline[i-1] = triple.second; f1Baseline[i-1] = triple.third;
					LogInfo.end_track();
				}
				if(evaluateDev) {
					LogInfo.begin_track("Dev");
					predicted = inferer.Infer(split.GetTestExamples(i), param, featureFactory);
					triple = Scorer.score(predicted);
					precisionDev[i-1] = triple.first; recallDev[i-1] = triple.second; f1Dev[i-1] = triple.third;
					LogInfo.end_track();
				}
				LogInfo.end_track();
	
				if(useOneLoop)
					break;
				LogInfo.end_track();
			}
			LogInfo.end_track();
			
			LogInfo.begin_track("Evaluation");
			if(!useSmallSample) {
				if(evaluateTrain) {
					LogInfo.begin_track("Training");
					printScores("Train", precisionTrain, recallTrain, f1Train);
					LogInfo.end_track();
				}
				if(evaluateBaseline) {
					LogInfo.begin_track("Baseline");
					printScores("Baseline", precisionBaseline, recallBaseline, f1Baseline);
					LogInfo.end_track();
				}
				if(evaluateDev) {
					LogInfo.begin_track("dev");
					printScores("Dev", precisionDev, recallDev, f1Dev);
					LogInfo.end_track();
				}
			}
			LogInfo.end_track();
		}
	}

	public void printScores(String category, double[] precision, double[] recall, double[] f1) {
		LogInfo.logs(String.format("\n------------------------------------------" + category + "-------------------------------------------"));
		LogInfo.logs(printScore("Precision", precision));
		LogInfo.logs(printScore("Recall", recall));
		LogInfo.logs(printScore("F1", f1));
	}

	public String printScore(String scoreType, double[] scores) {
		StringBuilder ret = new StringBuilder(String.format("%s ", scoreType));
		for(double s:scores)
			ret.append(String.format("%.3f ", s));
		ret.append(String.format("%.3f", getAverage(scores)));
		return ret.toString();
	}

	public double getAverage(double[] scores) {
		double sum = 0;
		for(double d:scores)
			sum+=d;
		return sum/scores.length;
	}

	/***
	 * Entry point to the bio process project. 
	 * @param args
	 */
	public static void main(String[] args) {
		Execution.run(args,
				"Main", new Main());
	}

	@Override
	public void run() {
		
		LogInfo.begin_track("main");
		Properties props = StringUtils.propFileToProperties(propertyFile);
		String trainDirectory = props.getProperty("train.dir"), testDirectory = props.getProperty("test.dir"),
				sampleDirectory = props.getProperty("sample.dir");

		HashMap<String, String> folders = new HashMap<String, String>();
		folders.put("test", testDirectory);
		folders.put("train", trainDirectory);
		folders.put("sample", sampleDirectory);

		if(mode.equals("entity")) {
			LogInfo.logs("Running entity prediction");
			new Main().runEntityPrediction(folders);
		}
		else if(mode.equals("entitygold")) {
			LogInfo.logs("Running entity prediction from GOLD events");
			new Main().runPrediction(folders, new EntityFeatureFactory(), new EntityPredictionLearner(), new EntityPredictionInferer(), new Scorer());
		}
		else if(mode.equals("entitystandalone")) {
			LogInfo.logs("Running entity standalone");
			new Main().runEntityStandalonePrediction(folders);
		}
		else if(mode.equals("eventstandalone")) {
			LogInfo.logs("Running event prediction");
			new Main().runEventStandalonePrediction(folders);
		}
		else if(mode.equals("eventgold")) {
			LogInfo.logs("Running event prediction with GOLD entities");
			new Main().runPrediction(folders, new EventExtendedFeatureFactory(), new EventPredictionLearner(), new EventPredictionInferer(), new Scorer());
		}
		else if(mode.equals("event")) {
			LogInfo.logs("Running event prediction");
			new Main().runEventPrediction(folders);
		}
		else if(mode.equals("io")) {
			LogInfo.logs("Running iterative optimization");
			new Main().runIterativeOptimization(folders);
		}
		else if(mode.equals("srl")) {
			LogInfo.logs("Running SRL");
			new Main().runSRLPrediction(folders);
		}
		LogInfo.end_track();
	}

	private void runSRLPrediction(HashMap<String, String> folders) {
		int NumCrossValidation = 10;
		boolean small = true;
		BioprocessDataset dataset = loadDataSet(folders, small, false);
		SRLFeatureFactory featureFactory = new SRLFeatureFactory();
		SRLPredictionLearner learner = new SRLPredictionLearner();
		SRLPredictionInferer inferer = new SRLPredictionInferer(); 
		Scorer scorer = new Scorer();
		if (small) {
			Params param = learner.learn(dataset.examples("sample"), featureFactory);
			//featureFactory = new SRLFeatureFactory(param.labelIndex);
			List<BioDatum> predicted = inferer.Infer(dataset.examples("sample"), param, featureFactory);
			Triple<Double, Double, Double> triple = Scorer.score(predicted);
			LogInfo.logs("Precision : " + triple.first);
			LogInfo.logs("Recall    : " + triple.second);
			LogInfo.logs("F1 score  : " + triple.third);
		}
		else {
			CrossValidationSplit split = new CrossValidationSplit(dataset.examples("train"), NumCrossValidation);
			double[] precisionBaseline = new double[NumCrossValidation], recallBaseline = new double[NumCrossValidation], f1Baseline = new double[NumCrossValidation];
			for(int i = 1; i <= NumCrossValidation; i++) {
				LogInfo.begin_track("Iteration " + i);
				Params param = learner.learn(split.GetTrainExamples(i), featureFactory);
				//featureFactory = new SRLFeatureFactory(param.labelIndex);
				List<BioDatum> predicted = inferer.Infer(split.GetTestExamples(i), param, featureFactory);
				Triple<Double, Double, Double> triple = Scorer.score(predicted);
				precisionBaseline[i-1] = triple.first; recallBaseline[i-1] = triple.second; f1Baseline[i-1] = triple.third;
				LogInfo.end_track();
				//break;
			}
			printScores("Dev", precisionBaseline, recallBaseline, f1Baseline);
		}
	}

	private void runEntityStandalonePrediction(HashMap<String, String> folders) {
		int NumCrossValidation = 10;
		IterativeOptimizer opt = new IterativeOptimizer();
		BioprocessDataset dataset = loadDataSet(folders, false, false);
		CrossValidationSplit split = new CrossValidationSplit(dataset.examples("train"), NumCrossValidation);
		double[] precisionDev = new double[NumCrossValidation], recallDev = new double[NumCrossValidation], f1Dev = new double[NumCrossValidation];
		for(int i = 1; i <= NumCrossValidation; i++) {
			LogInfo.begin_track("Iteration " + i);
			
			Learner entityLearner = new EntityPredictionLearner();
			FeatureExtractor entityFeatureFactory = new EntityStandaloneFeatureFactory();

			Inferer entityInferer = new EntityStandaloneInferer();
			Params entityStandaloneParams = entityLearner.learn(split.GetTrainExamples(i), entityFeatureFactory);
			List<BioDatum> predictedEntities = entityInferer.Infer(split.GetTestExamples(i), entityStandaloneParams, entityFeatureFactory);
			Triple<Double, Double, Double> entityTriple = Scorer.scoreEntities(split.GetTestExamples(i), predictedEntities);

			precisionDev[i-1] = entityTriple.first; recallDev[i-1] = entityTriple.second; f1Dev[i-1] = entityTriple.third;
			LogInfo.end_track();
			//break;
		}
		printScores("Dev", precisionDev, recallDev, f1Dev);
	}
	
	private void runEntityPrediction(HashMap<String, String> folders) {
		int NumCrossValidation = 10;
		BioprocessDataset dataset = loadDataSet(folders, false, false);
		CrossValidationSplit split = new CrossValidationSplit(dataset.examples("train"), NumCrossValidation);
		double[] precisionDev = new double[NumCrossValidation], recallDev = new double[NumCrossValidation], f1Dev = new double[NumCrossValidation];

		for(int i = 1; i <= NumCrossValidation; i++) {
			LogInfo.begin_track("Iteration " + i);
			
			Learner eventLearner = new EventPredictionLearner();
			FeatureExtractor eventFeatureFactory = new EventFeatureFactory();
			Inferer eventInferer = new EventPredictionInferer();
			Params eventParam = eventLearner.learn(split.GetTrainExamples(i), eventFeatureFactory);
			List<BioDatum> predicted = eventInferer.Infer(split.GetTestExamples(i), eventParam, eventFeatureFactory);
			
			Learner entityLearner = new EntityPredictionLearner();
			FeatureExtractor entityFeatureFactory = new EntityFeatureFactory();
			Params entityParam = entityLearner.learn(split.GetTrainExamples(i), entityFeatureFactory);
			EntityPredictionInferer entityInferer = new EntityPredictionInferer(predicted);
			
			List<BioDatum> entityPredicted = entityInferer.Infer(split.GetTestExamples(i), entityParam, entityFeatureFactory);
			Triple<Double, Double, Double> triple = Scorer.scoreEntities(split.GetTestExamples(i), entityPredicted);
			precisionDev[i-1] = triple.first; recallDev[i-1] = triple.second; f1Dev[i-1] = triple.third;
			LogInfo.end_track();
		}
		printScores("Dev", precisionDev, recallDev, f1Dev);
	}
	
	private void runEventStandalonePrediction(HashMap<String, String> folders) {
		int NumCrossValidation = 10;
		BioprocessDataset dataset = loadDataSet(folders, false, false);
		CrossValidationSplit split = new CrossValidationSplit(dataset.examples("train"), NumCrossValidation);
		double[] precisionDev = new double[NumCrossValidation], recallDev = new double[NumCrossValidation], f1Dev = new double[NumCrossValidation];

		for(int i = 1; i <= NumCrossValidation; i++) {
			LogInfo.begin_track("Iteration " + i);
			
			EventPredictionLearner eventLearner = new EventPredictionLearner();
			EventFeatureFactory eventFeatureFactory = new EventFeatureFactory();
			EventPredictionInferer eventInferer = new EventPredictionInferer();
			Params eventParam = eventLearner.learn(split.GetTrainExamples(i), eventFeatureFactory);
			List<BioDatum> result = eventInferer.Infer(split.GetTestExamples(i), eventParam, eventFeatureFactory);

			Triple<Double, Double, Double> triple = Scorer.scoreEvents(split.GetTestExamples(i), result);
			precisionDev[i-1] = triple.first; recallDev[i-1] = triple.second; f1Dev[i-1] = triple.third;
			LogInfo.end_track();
		}
		printScores("Dev", precisionDev, recallDev, f1Dev);
	}
	
	private void runEventPrediction(HashMap<String, String> folders) {
		int NumCrossValidation = 10;
		BioprocessDataset dataset = loadDataSet(folders, false, false);
		CrossValidationSplit split = new CrossValidationSplit(dataset.examples("train"), NumCrossValidation);
		double[] precisionDev = new double[NumCrossValidation], recallDev = new double[NumCrossValidation], f1Dev = new double[NumCrossValidation];

		for(int i = 1; i <= NumCrossValidation; i++) {
			LogInfo.begin_track("Iteration " + i);
			
			Learner entityLearner = new EntityPredictionLearner();
			FeatureExtractor entityFeatureFactory = new EntityStandaloneFeatureFactory();
			Inferer entityInferer = new EntityStandaloneInferer();
			Params entityStandaloneParams = entityLearner.learn(split.GetTrainExamples(i), entityFeatureFactory);
			List<BioDatum> predictedEntities = entityInferer.Infer(split.GetTestExamples(i), entityStandaloneParams, entityFeatureFactory);
			
			Learner eventLearner = new EventPredictionLearner();
			FeatureExtractor eventFeatureFactory = new EventExtendedFeatureFactory();
			Inferer eventInferer = new EventPredictionInferer(predictedEntities);
			Params eventParam = eventLearner.learn(split.GetTrainExamples(i), eventFeatureFactory);
			List<BioDatum> eventPredicted = eventInferer.Infer(split.GetTestExamples(i), eventParam, eventFeatureFactory);
			
			
			Triple<Double, Double, Double> triple = Scorer.scoreEvents(split.GetTestExamples(i), eventPredicted);
			precisionDev[i-1] = triple.first; recallDev[i-1] = triple.second; f1Dev[i-1] = triple.third;
			LogInfo.end_track();
		}
		printScores("Dev", precisionDev, recallDev, f1Dev);
	}

	private void runIterativeOptimization(HashMap<String, String> folders) {
		int NumCrossValidation = 10;
		IterativeOptimizer opt = new IterativeOptimizer();
		BioprocessDataset dataset = loadDataSet(folders, false, false);
		CrossValidationSplit split = new CrossValidationSplit(dataset.examples("train"), NumCrossValidation);
		double[] precisionTrigger = new double[NumCrossValidation], recallTrigger = new double[NumCrossValidation], f1Trigger = new double[NumCrossValidation];
		double[] precisionEntity = new double[NumCrossValidation], recallEntity = new double[NumCrossValidation], f1Entity = new double[NumCrossValidation];
		for(int i = 1; i <= NumCrossValidation; i++) {
			LogInfo.begin_track("Iteration " + i);
			Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> triple = opt.optimize(split.GetTrainExamples(i), split.GetTestExamples(i));
			precisionTrigger[i-1] = triple.first.first; recallTrigger[i-1] = triple.first.second; f1Trigger[i-1] = triple.first.third;
			precisionEntity[i-1] = triple.second.first; recallEntity[i-1] = triple.second.second; f1Entity[i-1] = triple.second.third;
			LogInfo.end_track();
			//break;
		}
		printScores("Dev Trigger", precisionTrigger, recallTrigger, f1Trigger);
		printScores("Dev Entity", precisionEntity, recallEntity, f1Entity);
	}
	
	private BioprocessDataset loadDataSet(HashMap<String, String> groups, boolean useSmallSample, boolean refreshDataFile) {
		String examplesFileName = "data.bpa";
		BioprocessDataset dataset = new BioprocessDataset(groups);

		if(!useSmallSample) {
			File f = new File(examplesFileName);
			if(f.exists() && !refreshDataFile) {
				LogInfo.begin_track("Quick data read");
				dataset.allExamples.put("train", Utils.readFile(examplesFileName));
				LogInfo.end_track();
			}
			else {
				dataset.read("train");
				Utils.writeFile(dataset.examples("train"), examplesFileName);
			}
		}
		else{
			dataset.read("sample");
		}
		return dataset;
	}
}
