package edu.stanford.nlp.bioprocess;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;

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
		boolean evaluateTrain = true, evaluateBaseline = true, evaluateDev = true;
		//Flags to control sample on which test is to be run. useSmallSample runs on 2 sample files, while useOneLoop runs one fold of CV.
		//refreshDataFile is to re-generate the bpa (bio process annotation) file
		boolean useSmallSample = false, useOneLoop = false, refreshDataFile = false;
		//useSmallSample = true;
		//useOneLoop = true;
		//refreshDataFile = true;
		String examplesFileName = "data.bpa";
		BioprocessDataset dataset = new BioprocessDataset(groups);
		CrossValidationSplit split = null;


		if(!useSmallSample) {
			File f = new File(examplesFileName);
			if(f.exists() && !refreshDataFile)
				dataset.allExamples.put("train", Utils.readFile(examplesFileName));
			else {
				dataset.read("train");
				Utils.writeFile(dataset.examples("train"), examplesFileName);
			}
			split = new CrossValidationSplit((ArrayList<Example>) dataset.examples("train"), NumCrossValidation);
		}
		else{
			dataset.read("sample");
		}
			
		for(int i = 1; i <= NumCrossValidation; i++) {
			if(useSmallSample) {
				Params param = learner.learn(dataset.examples("sample"), featureFactory);
				List<Datum> predicted = inferer.Infer(dataset.examples("sample"), param);
				Triple<Double, Double, Double> triple = Scorer.score(predicted);
				LogInfo.logs("Precision : " + triple.first);
				LogInfo.logs("Recall    : " + triple.second);
				LogInfo.logs("F1 score  : " + triple.third);
				break;
			}
			else {
				LogInfo.logs("Iteration: "+i);
				Params param = learner.learn(split.GetTrainExamples(i), featureFactory);
				List<Datum> predicted;
				Triple<Double, Double, Double> triple;
				if(evaluateTrain) {
					predicted = inferer.Infer(split.GetTrainExamples(i), param);
					triple = Scorer.score(predicted);
					precisionTrain[i-1] = triple.first; recallTrain[i-1] = triple.second; f1Train[i-1] = triple.third;
				}
				if(evaluateBaseline) {
					predicted = inferer.BaselineInfer(split.GetTestExamples(i), param);
					triple = Scorer.score(predicted);
					precisionBaseline[i-1] = triple.first; recallBaseline[i-1] = triple.second; f1Baseline[i-1] = triple.third;
				}
				if(evaluateDev) {
					predicted = inferer.Infer(split.GetTestExamples(i), param);
					triple = Scorer.score(predicted);
					precisionDev[i-1] = triple.first; recallDev[i-1] = triple.second; f1Dev[i-1] = triple.third;
				}

			}
			if(useOneLoop)
				break;
		}
		if(!useSmallSample) {
			if(evaluateTrain) {
				printScores("Train", precisionTrain, recallTrain, f1Train);
			}
			if(evaluateBaseline) {
				printScores("Baseline", precisionBaseline, recallBaseline, f1Baseline);
			}
			if(evaluateDev) {
				printScores("Dev", precisionDev, recallDev, f1Dev);
			}
		}
	}

	public void printScores(String category, double[] precision, double[] recall, double[] f1) {
		LogInfo.logs(String.format("\n------------------------------------------" + category + "-------------------------------------------"));
		LogInfo.logs(printScore("Precision", precision));
		LogInfo.logs(printScore("Recall", recall));
		LogInfo.logs(printScore("F1", f1));
	}

	public String printScore(String scoreType, double[] scores) {
		StringBuilder ret = new StringBuilder(String.format("%-15s: ", scoreType));
		for(double s:scores)
			ret.append(String.format("%.3f ", s));
		ret.append(String.format(" Average : %.3f", getAverage(scores)));
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

		
		if(mode.equals("entity")) 
			new Main().runPrediction(folders, new EntityFeatureFactory(), new EntityPredictionLearner(), new EntityPredictionInferer(), new Scorer());
		if(mode.equals("event"))
			new Main().runPrediction(folders, new EventFeatureFactory(), new EventPredictionLearner(), new EventPredictionInferer(), new Scorer());
		if(mode.equals("em"))
			new Main().runPrediction(folders, new EventExtendedFeatureFactory(), new EventPredictionLearner(), new EventPredictionInferer(), new Scorer());
		LogInfo.end_track();
	}
}
