package edu.stanford.nlp.bioprocess;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EventMentionsAnnotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;

public class Main implements Runnable {
	public static final String ANSI_RESET = "\u001B[0m";
	public static final String ANSI_BLACK = "\u001B[30m";
	public static final String ANSI_RED = "\u001B[31m";
	public static final String ANSI_GREEN = "\u001B[32m";
	public static final String ANSI_YELLOW = "\u001B[33m";
	public static final String ANSI_BLUE = "\u001B[34m";
	public static final String ANSI_PURPLE = "\u001B[35m";
	public static final String ANSI_CYAN = "\u001B[36m";
	public static final String ANSI_WHITE = "\u001B[37m";
	
	boolean runPrevBaseline = false, runBetterBaseline = false, 
			runLocalBase = false, runLocalModel = false,
			runGlobalModel = false, runParameterSearch = false;
	boolean runEventRelationTest = true;
	static final String MODELS_DIRECTORY = "models/";
	double alpha1_ = 0.0, alpha2_ = 0.0, alpha3_ = 0.0, alpha4_ = 0.0, alpha5_ = 0.0, alpha6_ = 0.0, alpha7_ = 0.0;
	boolean connectedComponent_ = false, sameEvent_ = false, previousEvent_ = false, sameEventContradictions_ = false;
	final String GLOBAL_PARAM_FILE_NAME = "GlobalParameters.txt";
	final String GLOBAL_PARAM_FILE = MODELS_DIRECTORY + GLOBAL_PARAM_FILE_NAME;

	public static final String 
			        EVENT_RELATION_LOCAL_MODEL_FILE_NAME = "EventRelation_Local_model.ser",
					EVENT_RELATION_LOCALBASE_MODEL_FILE_NAME = "EventRelation_LocalBase_model.ser",
					EVENT_RELATION_GLOBAL_MODEL_FILE_NAME = "EventRelation_Global_model.ser",
					EVENT_STANDALONE_MODEL_FILE_NAME = "EventStandalone_model.ser",
					ENTITY_STANDALONE_MODEL_FILE_NAME =  "EntityStandalone_model.ser",
					EVENT_MODEL_FILE_NAME = "Event_model.ser",
					ENTITY_MODEL_FILE_NAME = "Entity_model.ser";
	
	public static final String 
	                EVENT_RELATION_LOCAL_MODEL = MODELS_DIRECTORY + "EventRelation_Local_model.ser",
					EVENT_RELATION_LOCALBASE_MODEL = MODELS_DIRECTORY + "EventRelation_LocalBase_model.ser",
					EVENT_RELATION_GLOBAL_MODEL = MODELS_DIRECTORY + "EventRelation_Global_model.ser",
					EVENT_STANDALONE_MODEL = MODELS_DIRECTORY +"EventStandalone_model.ser",
					ENTITY_STANDALONE_MODEL = MODELS_DIRECTORY + "EntityStandalone_model.ser",
					EVENT_MODEL = MODELS_DIRECTORY + "Event_model.ser",
					ENTITY_MODEL = MODELS_DIRECTORY + "Entity_model.ser";

	@Option(gloss="The running mode: event, entity, or em") public String mode = "result";
	@Option(gloss="Dataset dir") public String datasetDir;
	@Option(gloss="Should we include lexical features?") public boolean useLexicalFeatures = true;
	@Option(gloss="Run on dev or test") public String runOn;
	@Option(gloss="Model to run") public String runModel;	

	public void runPrediction(HashMap<String, String> groups, FeatureExtractor featureFactory, Learner learner, Inferer inferer, Scorer scorer) {
		int NumCrossValidation = 10;
		double[] precisionTrain = new double[NumCrossValidation], recallTrain = new double[NumCrossValidation], f1Train = new double[NumCrossValidation], 
				precisionDev = new double[NumCrossValidation], recallDev = new double[NumCrossValidation], f1Dev = new double[NumCrossValidation],
				precisionBaseline = new double[NumCrossValidation], recallBaseline = new double[NumCrossValidation], f1Baseline = new double[NumCrossValidation];

		//Flags to indicate if evaluation of model should be run on training set, baseline and dev-test set.
		boolean evaluateTrain = false, evaluateBaseline = false, evaluateDev = true;
		//Flags to control sample on which test is to be run. useSmallSample runs on 2 sample files, while useOneLoop runs one fold of CV.
		//refreshDataFile is to re-generate the bpa (bio process annotation) file
		boolean useSmallSample = false, useOneLoop = false, refreshDataFile = false;
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
		LogInfo.logs(printScore("Recall   ", recall));
		LogInfo.logs(printScore("F1       ", f1));
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
		String trainDirectory = datasetDir+"/train/";
		String testDirectory = datasetDir+"/test/";
		String sampleDirectory = datasetDir+"/sample/";
		
		HashMap<String, String> folders = new HashMap<String, String>();
		folders.put("test", testDirectory);
		folders.put("train", trainDirectory);
		folders.put("sample", sampleDirectory);
		
		if(mode != null) {
			mode = mode.toLowerCase();
		}
		if(runModel != null) {
			runModel = runModel.toLowerCase();
		}
		if(runOn != null) {
			runOn = runOn.toLowerCase();
		}

		if(mode.equals("entity")) {
			LogInfo.logs("Running entity prediction");
			runEntityPrediction(folders);
		}
		else if(mode.equals("entitygold")) {
			LogInfo.logs("Running entity prediction from GOLD events");
			runPrediction(folders, new EntityFeatureFactory(useLexicalFeatures), new Learner(), new EntityPredictionInferer(), new Scorer());
		}
		else if(mode.equals("entitystandalone")) {
			LogInfo.logs("Running entity standalone");
			runEntityStandalonePrediction(folders);
		}
		else if(mode.equals("eventstandalone")) {
			LogInfo.logs("Running event standalone prediction");
			runEventStandalonePrediction(folders);
		}
		else if(mode.equals("eventgold")) {
			LogInfo.logs("Running event prediction with GOLD entities");
			runPrediction(folders, new EventExtendedFeatureFactory(useLexicalFeatures), new Learner(), new EventPredictionInferer(), new Scorer());
		}
		else if(mode.equals("event")) {
			LogInfo.logs("Running event prediction");
			runEventPrediction(folders);
		}
		else if(mode.equals("io")) {
			LogInfo.logs("Running iterative optimization");
			runIterativeOptimization(folders);
		}
		else if(mode.equals("srl")) {
			LogInfo.logs("Running SRL");
			runSRLPrediction(folders);
		}
		else if(mode.equals("all")) {
			LogInfo.logs("Run all");
			runAll(folders);
		}
		else if(mode.equals("eventrelation")) {
			LogInfo.logs("Running event relation");
			//computeStats(folders);
			
			if(!runEventRelationTest)
				runEventRelationsPrediction(folders);
			else
				runEventRelationsPredictionTest(folders);
			
			//Utils.getEquivalentTriples(new Triple<String, String, String>("PreviousEvent", "SuperEvent", "Causes"));
		}
		else if(mode.equals("pipeline")) {
			if (runOn != null && runModel != null &&
					!runOn.isEmpty() && !runModel.isEmpty()) {
				System.out.println("Invalid parameters. 'runon' and 'runmodel' are not required for pipeline mode.");
			}
			else {
				runModel = "global";
				runPipelinePrediction(folders);
			}
			
		}
		else if(mode.equals("interactive")) {
			LogInfo.logs("Running interactive mode.");
			runInteractiveMode(folders);
		}
		else if(mode.equals("result")) {
			if (!runOn.equals("dev") && !runOn.equals("test")) {
				System.out.println("Invalid dataset provided. Choose 'dev' or 'test'");
			}
			else if (!runModel.equals("baseline") && !runModel.equals("chain") &&
					 !runModel.equals("localbase") && !runModel.equals("local") &&
					 !runModel.equals("global")) {
				System.out.println("Invalid model provided. Choose 'baseline', 'localbase', 'local', 'chain' or 'global'.");
			}
			else {
				if(runModel.equals("baseline")) {
					runPrevBaseline = true;
				}
				else if(runModel.equals("chain")) {
					runBetterBaseline = true;
				}
				else if(runModel.equals("localbase")) {
					runLocalBase = true;
				}
				else if(runModel.equals("local")) {
					runLocalModel = true;
				}
				else if(runModel.equals("global")) {
					runGlobalModel = true;
					runParameterSearch = true;
				}
				if(runOn.equals("dev")) {
					runEventRelationsPrediction(folders);
				}
				else if(runOn.equals("test")) {
					runEventRelationsPredictionTest(folders);
				}
			}
		}
		LogInfo.end_track();
	}

	private void runInteractiveMode(HashMap<String, String> folders) {
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
		runModel = "global";
		
		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
		
		BioProcessFormatReader bioReader = new BioProcessFormatReader();
		StanfordCoreNLP  processor = new StanfordCoreNLP(props, false);
		bioReader.setProcessor(processor);

		System.out.println(ANSI_GREEN + "\n\nBegin by entering your input paragraph at prompt.");
		System.out.println(ANSI_GREEN + "Type 'q' or 'quit' to exit.");
		while(true) {
			try {
				System.out.print(ANSI_RESET + "\n>");
				String input = reader.readLine();
				if(input.equals("q") ||
						input.equals("quit")) {
					break;
				}
				Example ex = bioReader.createAnnotationFromString(input);
				
				List<Example> examples = new ArrayList<Example>();
				examples.add(ex);
				
				IterativeOptimizer optimizer = new IterativeOptimizer();
				optimizer.runInference(examples);
			}
			catch (Exception ex) {
				System.out.println(ANSI_RED + "Sorry, could not run event extraction on the given input." + ANSI_RESET);
			}
		}
		System.out.println(ANSI_RESET);
	}

	private void runPipelinePrediction(HashMap<String, String> folders) {
		Utils.clearFolderContent("GraphViz");
		BioprocessDataset trainDataset = loadDataSet(folders, false, false);
		
		BioprocessDataset testDataset = loadTestDataSet(folders, false);
		
		IterativeOptimizer opt = new IterativeOptimizer();
		
		Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> pairTriple;// = Scorer.scoreEventRelations(result);
		
		pairTriple = opt.runPipelinePrediction(trainDataset.examples("train"), testDataset.examples("test"), true, runModel);
		
		LogInfo.logs("Full Micro precision");
		LogInfo.logs("P : " + String.format("%.4f", pairTriple.first.first));
		LogInfo.logs("R : " + String.format("%.4f", pairTriple.first.second));
		LogInfo.logs("F : " + String.format("%.4f", pairTriple.first.third));
	}

	private void runAll(HashMap<String, String> folders) {
		int NumCrossValidation = 10;
		BioprocessDataset dataset = loadDataSet(folders, false, false);
		CrossValidationSplit split = new CrossValidationSplit(dataset.examples("train"), NumCrossValidation);
		double[] precisionEntBasic = new double[NumCrossValidation], recallEntBasic = new double[NumCrossValidation], f1EntBasic = new double[NumCrossValidation];
		double[] precisionEvtBasic = new double[NumCrossValidation], recallEvtBasic = new double[NumCrossValidation], f1EvtBasic = new double[NumCrossValidation];
		double[] precisionEntIO = new double[NumCrossValidation], recallEntIO = new double[NumCrossValidation], f1EntIO = new double[NumCrossValidation];
		double[] precisionEvtIO = new double[NumCrossValidation], recallEvtIO = new double[NumCrossValidation], f1EvtIO = new double[NumCrossValidation];
		IterativeOptimizer opt = new IterativeOptimizer();
		
		for(int i = 1; i <= NumCrossValidation; i++) {
			LogInfo.begin_track("Iteration " + i);
			
			Learner eventLearner = new Learner();
			EventFeatureFactory eventFeatureFactory = new EventFeatureFactory(useLexicalFeatures);
			EventPredictionInferer eventInferer = new EventPredictionInferer();
			Params eventParam = eventLearner.learn(split.GetTrainExamples(i), eventFeatureFactory);
			List<BioDatum> result = eventInferer.Infer(split.GetTestExamples(i), eventParam, eventFeatureFactory);

			Triple<Double, Double, Double> triple = Scorer.scoreEvents(split.GetTestExamples(i), result);
			precisionEvtBasic[i-1] = triple.first; recallEvtBasic[i-1] = triple.second; f1EvtBasic[i-1] = triple.third;
			
			Learner entityLearner = new Learner();
			FeatureExtractor entityFeatureFactory = new EntityFeatureFactory(useLexicalFeatures);
			Params entityParam = entityLearner.learn(split.GetTrainExamples(i), entityFeatureFactory);
			EntityPredictionInferer entityInferer = new EntityPredictionInferer(result);
			
			List<BioDatum> entityPredicted = entityInferer.Infer(split.GetTestExamples(i), entityParam, entityFeatureFactory);
			triple = Scorer.scoreEntities(split.GetTestExamples(i), entityPredicted);
			precisionEntBasic[i-1] = triple.first; recallEntBasic[i-1] = triple.second; f1EntBasic[i-1] = triple.third;
			
			Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> pairTriple = opt.optimize(split.GetTrainExamples(i), split.GetTestExamples(i), useLexicalFeatures);
			precisionEvtIO[i-1] = pairTriple.first.first; recallEvtIO[i-1] = pairTriple.first.second; f1EvtIO[i-1] = pairTriple.first.third;
			precisionEntIO[i-1] = pairTriple.second.first; recallEntIO[i-1] = pairTriple.second.second; f1EntIO[i-1] = pairTriple.second.third;
			
			LogInfo.end_track();
		}
		printScores("Event Basic", precisionEvtBasic, recallEvtBasic, f1EvtBasic);
		printScores("Entity Basic", precisionEntBasic, recallEntBasic, f1EntBasic);
		printScores("Event IO", precisionEvtIO, recallEvtIO, f1EvtIO);
		printScores("Entity IO", precisionEntIO, recallEntIO, f1EntIO);
	}

	private void runSRLPrediction(HashMap<String, String> folders) {
		int NumCrossValidation = 10;
		boolean small = false;
		BioprocessDataset dataset = loadDataSet(folders, small, false);
		SRLFeatureFactory featureFactory = new SRLFeatureFactory(useLexicalFeatures);
		Learner learner = new Learner();
		SRLPredictionInferer inferer = new SRLPredictionInferer(); 
		if (small) {
			Params param = learner.learn(dataset.examples("sample"), featureFactory);
			//featureFactory = new SRLFeatureFactory(param.labelIndex);
			List<BioDatum> predicted = inferer.BaselineInfer(dataset.examples("sample"), param, featureFactory);
			Triple<Double, Double, Double> triple = Scorer.scoreSRL(dataset.examples("sample"), predicted);
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
				Triple<Double, Double, Double> triple = Scorer.scoreSRL(split.GetTestExamples(i), predicted);
				precisionBaseline[i-1] = triple.first; recallBaseline[i-1] = triple.second; f1Baseline[i-1] = triple.third;
				LogInfo.end_track();
				//break;
			}
			printScores("Dev", precisionBaseline, recallBaseline, f1Baseline);
		}
	}

	private void runEntityStandalonePrediction(HashMap<String, String> folders) {
		int NumCrossValidation = 10;
		BioprocessDataset dataset = loadDataSet(folders, false, false);
		CrossValidationSplit split = new CrossValidationSplit(dataset.examples("train"), NumCrossValidation);
		double[] precisionDev = new double[NumCrossValidation], recallDev = new double[NumCrossValidation], f1Dev = new double[NumCrossValidation];
		for(int i = 1; i <= NumCrossValidation; i++) {
			LogInfo.begin_track("Iteration " + i);
			
			Learner entityLearner = new Learner();
			FeatureExtractor entityFeatureFactory = new EntityStandaloneFeatureFactory(useLexicalFeatures);

			Inferer entityInferer = new EntityStandaloneInferer();
			Params entityStandaloneParams = entityLearner.learn(split.GetTrainExamples(i), entityFeatureFactory);
			List<BioDatum> predictedEntities = entityInferer.Infer(split.GetTestExamples(i), entityStandaloneParams, entityFeatureFactory);
			Triple<Double, Double, Double> entityTriple = Scorer.score(predictedEntities);

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
			
			Learner eventLearner = new Learner();
			FeatureExtractor eventFeatureFactory = new EventFeatureFactory(useLexicalFeatures);
			Inferer eventInferer = new EventPredictionInferer();
			Params eventParam = eventLearner.learn(split.GetTrainExamples(i), eventFeatureFactory);
			List<BioDatum> predicted = eventInferer.Infer(split.GetTestExamples(i), eventParam, eventFeatureFactory);
			
			Learner entityLearner = new Learner();
			FeatureExtractor entityFeatureFactory = new EntityFeatureFactory(useLexicalFeatures);
			Params entityParam = entityLearner.learn(split.GetTrainExamples(i), entityFeatureFactory);
			EntityPredictionInferer entityInferer = new EntityPredictionInferer(predicted);
			
			List<BioDatum> entityPredicted = entityInferer.Infer(split.GetTestExamples(i), entityParam, entityFeatureFactory);
			//List<BioDatum> entityPredicted = entityInferer.BaselineInfer(split.GetTestExamples(i), entityParam, entityFeatureFactory);
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
			
			Learner eventLearner = new Learner();
			EventFeatureFactory eventFeatureFactory = new EventFeatureFactory(useLexicalFeatures);
			EventPredictionInferer eventInferer = new EventPredictionInferer();
			Params eventParam = eventLearner.learn(split.GetTrainExamples(i), eventFeatureFactory);
			List<BioDatum> result = eventInferer.Infer(split.GetTestExamples(i), eventParam, eventFeatureFactory);
			//List<BioDatum> result = eventInferer.BaselineInfer(split.GetTestExamples(i), eventParam, eventFeatureFactory);

			Triple<Double, Double, Double> triple = Scorer.scoreEvents(split.GetTestExamples(i), result);
			precisionDev[i-1] = triple.first; recallDev[i-1] = triple.second; f1Dev[i-1] = triple.third;
			LogInfo.end_track();
		}
		printScores("Dev Event standalone prediction", precisionDev, recallDev, f1Dev);
	}
	
	private void runEventRelationsPredictionTest(HashMap<String, String> folders) {
		Utils.clearFolderContent("GraphViz");
		
		//BioprocessDataset trainDataset = loadDataSet(folders, false, false);
		
		BioprocessDataset testDataset = loadTestDataSet(folders, false);
		
		//Learner eventRelationLearner = new Learner();
		EventRelationFeatureFactory eventRelationFeatureFactory = new EventRelationFeatureFactory(useLexicalFeatures, runModel);
		EventRelationInferer inferer = new EventRelationInferer(runModel);
		
		Params eventParam;// = eventRelationLearner.learn(trainDataset.examples("train"), eventRelationFeatureFactory);
		if(runLocalModel || runBetterBaseline) {
			//Utils.writeFile(eventParam, EVENT_RELATION_LOCAL_MODEL);
			eventParam =  (Params) Utils.readObject(EVENT_RELATION_LOCAL_MODEL);
		}
		else if(runGlobalModel) {
			//Utils.writeFile(eventParam, EVENT_RELATION_GLOBAL_MODEL);
			eventParam =  (Params) Utils.readObject(EVENT_RELATION_GLOBAL_MODEL);
		}
		else{
			//Utils.writeFile(eventParam, EVENT_RELATION_LOCALBASE_MODEL);
			eventParam =  (Params) Utils.readObject(EVENT_RELATION_LOCALBASE_MODEL);
		}
		
		List<BioDatum> result = null;
		if(runLocalModel || runGlobalModel || runLocalBase) {
			loadGlobalParameterValues();
			result = inferer.Infer(testDataset.examples("test"), eventParam, eventRelationFeatureFactory, runModel,
					connectedComponent_, sameEvent_, previousEvent_, sameEventContradictions_, alpha1_, alpha2_, alpha3_, alpha4_, alpha5_, alpha6_, alpha7_);
		}
		else if(runPrevBaseline) {
			result = inferer.BaselineInfer(testDataset.examples("test"), eventParam, eventRelationFeatureFactory);
		}
		else if(runBetterBaseline) {
			LogInfo.logs("Running better baseline");
			result = inferer.BetterBaselineInfer(testDataset.examples("test"), eventParam, eventRelationFeatureFactory);
		}
		
		BufferedWriter writer;
		try {
			writer = new BufferedWriter(new FileWriter("Global.txt"));
		
			for(BioDatum d:result) {
				writer.write("G:" + d.label + "," + "P:" + d.guessLabel+"\n");
			}
			
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> pairTriple;// = Scorer.scoreEventRelations(result);
		
		pairTriple = Scorer.scoreEventRelations(result);
		
		LogInfo.logs("Full Micro precision (Across all, Average over processes)");
		LogInfo.logs("P : " + String.format("%.4f", pairTriple.first.first) + String.format(", %.4f", EventRelationInferer.avgProcessPrecisionFull));
		LogInfo.logs("R : " + String.format("%.4f", pairTriple.first.second) + String.format(", %.4f", EventRelationInferer.avgProcessRecallFull));
		LogInfo.logs("F : " + String.format("%.4f", pairTriple.first.third) + String.format(", %.4f", EventRelationInferer.avgProcessF1Full));
		
		pairTriple = Scorer.scoreEventRelationsCollapsed(result);
		
		LogInfo.logs("Collapsed Micro precision (Across all, Average over processes)");
		LogInfo.logs("P : " + String.format("%.4f", pairTriple.first.first) + String.format(", %.4f", EventRelationInferer.avgProcessPrecisionCollapsed));
		LogInfo.logs("R : " + String.format("%.4f", pairTriple.first.second) + String.format(", %.4f", EventRelationInferer.avgProcessRecallCollapsed));
		LogInfo.logs("F : " + String.format("%.4f", pairTriple.first.third) + String.format(", %.4f", EventRelationInferer.avgProcessF1Collapsed));
		
		pairTriple = Scorer.scoreEventRelationsStructure(result);
		
		LogInfo.logs("Structure Micro precision (Across all, Average over processes)");
		LogInfo.logs("P : " + String.format("%.4f", pairTriple.first.first) + String.format(", %.4f", EventRelationInferer.avgProcessPrecisionStructure));
		LogInfo.logs("R : " + String.format("%.4f", pairTriple.first.second) + String.format(", %.4f", EventRelationInferer.avgProcessRecallStructure));
		LogInfo.logs("F : " + String.format("%.4f", pairTriple.first.third) + String.format(", %.4f", EventRelationInferer.avgProcessF1Structure));	
	}
	
	private void runEventRelationsPrediction(HashMap<String, String> folders) {
		int NumCrossValidation = 10;
		boolean small = false;
		boolean performParameterSearch = runParameterSearch;
		//double[] paramValues = new double[]{0.5, 10};
		double[] paramValues = new double[]{0, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10};
		boolean[] paramValuesBool = new boolean[]{false, true};
		HashMap<Integer, String> constraintNames = new HashMap<Integer, String>();
		constraintNames.put(1, "Connectivity constraint : Hard");
		constraintNames.put(2, "Same event triad closure : Hard"); 
		constraintNames.put(3, "Previous event : Hard");
		constraintNames.put(4, "Same event contradiction : Hard");

		constraintNames.put(5, "Cotemporal traid closure : Soft, Penalize");
		constraintNames.put(6, "Same event triad closure : Soft, Reward");
		constraintNames.put(7, "Same event triad closure : Soft, Penalize");
		constraintNames.put(8, "Cotemporal traid closure : Soft, Reward"); 
		constraintNames.put(9, "Causes traid closure : Soft, Reward");
		constraintNames.put(10, "Chain constraint : Soft, penalize");

		//Clearing folder for visualization
		Utils.moveFolderContent("GraphViz", "GraphVizPrev");
		Utils.clearFolderContent("GraphViz");
		BioprocessDataset dataset = loadDataSet(folders, small, false);
		
		Learner eventRelationLearner = new Learner();
		EventRelationFeatureFactory eventRelationFeatureFactory = new EventRelationFeatureFactory(useLexicalFeatures, runModel);
		EventRelationInferer inferer = new EventRelationInferer(runModel);
		List<String> relations = ArgumentRelation.getEventRelations();
		double[][] confusionMatrix = new double[relations.size()][relations.size()];
		
		if(small) {
			Params param = eventRelationLearner.learn(dataset.examples("sample"), eventRelationFeatureFactory);
			List<BioDatum> predicted = inferer.Infer(dataset.examples("sample"), param, eventRelationFeatureFactory, runModel,
					false, false, false, false, 0.0,0.0,0.0, 0.0,0.0,0.0,0.0);
			//List<BioDatum> predicted = inferer.BaselineInfer(dataset.examples("sample"), param, eventRelationFeatureFactory);
			Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> pairTriple = Scorer.scoreEventRelations(predicted);
			Scorer.updateMatrix(confusionMatrix, predicted, relations);
			
			System.out.println(Utils.findEventRelationDistribution(dataset.examples("sample")));
			Utils.printConfusionMatrix(confusionMatrix, relations, "ConfusionMatrix.csv");
			
			LogInfo.logs("Precision : " + pairTriple.first.first);
			LogInfo.logs("Recall    : " + pairTriple.first.second);
			LogInfo.logs("F1 score  : " + pairTriple.first.third);
		}
		else {
			LogInfo.logs(Utils.findEventRelationDistribution(dataset.examples("train")));
			CrossValidationSplit split = new CrossValidationSplit(dataset.examples("train"), NumCrossValidation);

			if(performParameterSearch) {
				try{
					ArrayList<Integer> paramArray = new ArrayList<Integer>();
					paramArray.add(1);paramArray.add(2);paramArray.add(3);paramArray.add(4);paramArray.add(5);paramArray.add(6);
					paramArray.add(7);paramArray.add(8);paramArray.add(9);paramArray.add(10);
					double alpha1 = 0.0, alpha2 = 0.0, alpha3 = 0.0, alpha4 = 0.0, alpha5 = 0.0, alpha6 = 0.0, alpha7 = 0.0;
					boolean connectedComponent = false, sameEvent = false, previousEvent = false, sameEventContradictions = false;
					BufferedWriter writer = new BufferedWriter(new FileWriter("scores.txt"));
					double bestF1 = 0.00, overallBestF1 = 0.00;
					int bestConstraint = -1, numParams = paramArray.size();
					
					for(int iterationCount = 0; iterationCount < numParams; iterationCount++) {
						writer.write("Iteration " + (iterationCount + 1) + "\n");	
						bestConstraint = -1;
						double bestParamValue = -1;
						bestF1 = 0.0;
						boolean bestParamValueBool = false;
						for(int paramCount = 0; paramCount < numParams - iterationCount; paramCount++)  {
							int paramIndexInParamArray = paramArray.get(paramCount);
							//bestF1 = 0.0;
							writer.write("\tParam name : " + constraintNames.get(paramIndexInParamArray) + "\n");
							
							if(paramIndexInParamArray <= 4){
								boolean oldValue = false;
								switch (paramIndexInParamArray) {
								case 1:
									oldValue = connectedComponent;
									break;
								case 2:
									oldValue = sameEvent;
									break;
								case 3:
									oldValue = previousEvent;
									break;
								case 4:
									oldValue = sameEventContradictions;
									break;
								}
								for(int paramValueCount = 0; paramValueCount < paramValuesBool.length; paramValueCount++) {
									switch (paramIndexInParamArray) {
									case 1:
										connectedComponent = paramValuesBool[paramValueCount];
										break;
									case 2:
										sameEvent = paramValuesBool[paramValueCount];
										break;
									case 3:
										previousEvent = paramValuesBool[paramValueCount];
										break;
									case 4:
										sameEventContradictions = paramValuesBool[paramValueCount];
										break;
									}
									
									//writer.write(String.format("Current values %b, %b, %b, %f, %f, %f\n", connectedComponent,
									//		sameEvent, previousEvent, alpha1, alpha2, alpha3));
									
									List<BioDatum> resultsFromAllFolds = new ArrayList<BioDatum>();
									
									for(int i = 1; i <= NumCrossValidation; i++) {
										Params eventParam = eventRelationLearner.learn(split.GetTrainExamples(i), eventRelationFeatureFactory);
										List<BioDatum> result = inferer.Infer(split.GetTestExamples(i), eventParam, eventRelationFeatureFactory, runModel,
												connectedComponent, sameEvent, previousEvent, sameEventContradictions, alpha1, alpha2, alpha3,
												alpha4, alpha5, alpha6, alpha7);
										
										resultsFromAllFolds.addAll(result);
									}
									
									Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> pairTriple = Scorer.scoreEventRelations(resultsFromAllFolds);
									writer.write(String.format("\t\tValue : %b, F1 : %f, P : %f, R : %f\n",  
											paramValuesBool[paramValueCount], pairTriple.first.third, pairTriple.first.first, pairTriple.first.second));
									LogInfo.logs("Micro precision " + paramIndexInParamArray + " " + paramValues[paramValueCount]);
									LogInfo.logs("Precision : " + pairTriple.first.first);
									LogInfo.logs("Recall    : " + pairTriple.first.second);
									LogInfo.logs("F1 score  : " + pairTriple.first.third);
									
									LogInfo.logs("\nMacro precision");
									LogInfo.logs("Precision : " + pairTriple.second.first);
									LogInfo.logs("Recall    : " + pairTriple.second.second);
									LogInfo.logs("F1 score  : " + pairTriple.second.third);
									
									if(pairTriple.first.third > bestF1) {
										bestF1 = pairTriple.first.third;
										bestParamValueBool = paramValuesBool[paramValueCount];
										bestConstraint = paramIndexInParamArray;
										writer.write("\t\tUpdating best F1. Best constraint: " + constraintNames.get(bestConstraint) +
															". Best value: " + bestParamValueBool+  "\n");
									}
									writer.flush();
								}
								//writer.write("\tBest value: " + bestParamValueBool + "\n");
								switch (paramIndexInParamArray) {
								case 1:
									connectedComponent = oldValue;
									break;
								case 2:
									sameEvent = oldValue;
									break;
								case 3:
									previousEvent = oldValue;
									break;
								case 4:
									sameEventContradictions = oldValue;
									break;
								}
							}
						
							else {
								double oldValue = 0.0;
								switch (paramIndexInParamArray) {
								case 5:
									oldValue = alpha1;
									break;
								case 6:
									oldValue = alpha2;
									break;
								case 7:
									oldValue = alpha3;
									break;
								case 8:
									oldValue = alpha4;
									break;
								case 9:
									oldValue = alpha5;
									break;
								case 10:
									oldValue = alpha7;
									break;
								}
								for(int paramValueCount = 0; paramValueCount < paramValues.length; paramValueCount++) {
									switch (paramIndexInParamArray) {
									case 5:
										alpha1 = paramValues[paramValueCount];
										break;
									case 6:
										alpha2 = paramValues[paramValueCount];
										break;
									case 7:
										alpha3 = paramValues[paramValueCount];
										break;
									case 8:
										alpha4 = paramValues[paramValueCount];
										break;
									case 9:
										alpha5 = paramValues[paramValueCount];
										break;
									case 10:
										alpha7 = paramValues[paramValueCount];
										break;
									}
									//writer.write(String.format("Current values %b, %b, %b, %f, %f, %f\n", connectedComponent,
									//		sameEvent, previousEvent, alpha1, alpha2, alpha3));
									if(alpha5 <= 1) {
										List<BioDatum> resultsFromAllFolds = new ArrayList<BioDatum>();
										
										for(int i = 1; i <= NumCrossValidation; i++) {
											Params eventParam = eventRelationLearner.learn(split.GetTrainExamples(i), eventRelationFeatureFactory);
											List<BioDatum> result = inferer.Infer(split.GetTestExamples(i), eventParam, eventRelationFeatureFactory, runModel,
													connectedComponent, sameEvent, previousEvent, sameEventContradictions, alpha1, alpha2, alpha3,
													alpha4, alpha5, alpha6, alpha7);
											
											resultsFromAllFolds.addAll(result);
										}
										
										Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> pairTriple = Scorer.scoreEventRelations(resultsFromAllFolds);
			
										writer.write(String.format("\t\tValue : %f, F1 : %f, P : %f, R : %f\n", 
													paramValues[paramValueCount], pairTriple.first.third, pairTriple.first.first, pairTriple.first.second));
										LogInfo.logs("Micro precision " + paramIndexInParamArray + " " + paramValues[paramValueCount]);
										LogInfo.logs("Precision : " + pairTriple.first.first);
										LogInfo.logs("Recall    : " + pairTriple.first.second);
										LogInfo.logs("F1 score  : " + pairTriple.first.third);
										
										LogInfo.logs("\nMacro precision");
										LogInfo.logs("Precision : " + pairTriple.second.first);
										LogInfo.logs("Recall    : " + pairTriple.second.second);
										LogInfo.logs("F1 score  : " + pairTriple.second.third);
										
										if(pairTriple.first.third > bestF1) {
											bestF1 = pairTriple.first.third;
											bestParamValue = paramValues[paramValueCount];
											bestConstraint = paramIndexInParamArray;
											writer.write("\t\tUpdating best F1. Best constraint: " + constraintNames.get(bestConstraint) +
													". Best value: " + bestParamValue+  "\n");
										}
									}
									writer.flush();
								}
								
								//writer.write("\tBest value: " + bestParamValue + "\n");
								
								switch (paramIndexInParamArray) {
								case 5:
									alpha1 = oldValue;
									break;
								case 6:
									alpha2 = oldValue;
									break;
								case 7:
									alpha3 = oldValue;
									break;
								case 8:
									alpha4 = oldValue;
									break;
								case 9:
									alpha5 = oldValue;
									break;
								case 10:
									alpha7 = oldValue;
									break;
								}
							}
							writer.write("--------------------------------------------------------\n");
						}
						writer.flush();
						if(bestF1 > overallBestF1) {
							writer.write("BestConstraint: " + constraintNames.get(bestConstraint) + ". Value: ");
							if(bestConstraint < 5)
								writer.write(((Boolean)bestParamValueBool).toString());
							else
								writer.write(((Double)bestParamValue).toString());
							writer.write("\n");
							overallBestF1 = bestF1;
							switch (bestConstraint) {
							case 5:
								alpha1 = bestParamValue;
								break;
							case 6:
								alpha2 = bestParamValue;
								break;
							case 7:
								alpha3 = bestParamValue;
								break;
							case 8:
								alpha4 = bestParamValue;
								break;
							case 9:
								alpha5 = bestParamValue;
								break;
							case 10:
								alpha7 = bestParamValue;
								break;
							case 1:
								connectedComponent = bestParamValueBool;
								break;
							case 2:
								sameEvent = bestParamValueBool;
								break;
							case 3:
								previousEvent = bestParamValueBool;
								break;
							case 4:
								sameEventContradictions = bestParamValueBool;
								break;
							}
							paramArray.remove(paramArray.indexOf(bestConstraint));
							writer.write("========================================================\n\n");
						}
						else {
							writer.write("No further improvement.");
							break;
						}
						writer.flush();
					}
					writer.close();
					BufferedWriter paramWriter = new BufferedWriter(new FileWriter(
							fig.exec.Execution.getActualExecDir() + "/" +  GLOBAL_PARAM_FILE_NAME));
					paramWriter.write(String.format("%s\t%s", "ConnectedComponent", connectedComponent));
					paramWriter.write(String.format("%s\t%s", "SameEvent", sameEvent));
					paramWriter.write(String.format("%s\t%s", "PreviousEvent", previousEvent));
					paramWriter.write(String.format("%s\t%s", "SameEventContradiction", sameEventContradictions));
					paramWriter.write(String.format("%s\t%s", "Alpha1", alpha1));
					paramWriter.write(String.format("%s\t%s", "Alpha2", alpha2));
					paramWriter.write(String.format("%s\t%s", "Alpha3", alpha3));
					paramWriter.write(String.format("%s\t%s", "Alpha4", alpha4));
					paramWriter.write(String.format("%s\t%s", "Alpha5", alpha5));
					paramWriter.write(String.format("%s\t%s", "Alpha6", alpha6));
					paramWriter.write(String.format("%s\t%s", "Alpha7", alpha7));
					paramWriter.close();
				}
				catch(Exception ex) {
					ex.printStackTrace();
					LogInfo.logs(ex.getMessage());
				}
			}
			else{
				List<BioDatum> resultsFromAllFolds = new ArrayList<BioDatum>();
				
				for(int i = 1; i <= NumCrossValidation; i++) {
					LogInfo.begin_track("Iteration " + i);
					
					Params eventParam = eventRelationLearner.learn(split.GetTrainExamples(i), eventRelationFeatureFactory);
					List<BioDatum> result = null;
					if(runLocalModel || runGlobalModel || runLocalBase)  {
						loadGlobalParameterValues();
						result = inferer.Infer(split.GetTestExamples(i), eventParam, eventRelationFeatureFactory, runModel,
								  connectedComponent_, sameEvent_, previousEvent_, sameEventContradictions_, alpha1_, alpha2_, alpha3_, alpha4_, alpha5_, alpha6_, alpha7_);
						          //true, false, false, false, 0.0,0.00,0,0,0.00,0.0,0.00);
					}
					else if(runPrevBaseline) {
						result = inferer.BaselineInfer(split.GetTestExamples(i), eventParam, eventRelationFeatureFactory);
					}
					else if(runBetterBaseline) {
						result = inferer.BetterBaselineInfer(split.GetTestExamples(i), eventParam, eventRelationFeatureFactory);
					}
					
					resultsFromAllFolds.addAll(result);

					Scorer.updateMatrix(confusionMatrix, result, relations);
									
					LogInfo.end_track();
				}
				
				Utils.printConfusionMatrix(confusionMatrix, relations, "out/ConfusionMatrix.csv");
				
				Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> pairTriple;
				
				pairTriple = Scorer.scoreEventRelations(resultsFromAllFolds);
				
				LogInfo.logs("Full Micro precision");
				LogInfo.logs("P : " + String.format("%.4f", pairTriple.first.first));
				LogInfo.logs("R : " + String.format("%.4f", pairTriple.first.second));
				LogInfo.logs("F : " + String.format("%.4f", pairTriple.first.third));
				
				pairTriple = Scorer.scoreEventRelationsCollapsed(resultsFromAllFolds);
				
				LogInfo.logs("Collapsed Micro precision");
				LogInfo.logs("P : " + String.format("%.4f", pairTriple.first.first));
				LogInfo.logs("R : " + String.format("%.4f", pairTriple.first.second));
				LogInfo.logs("F : " + String.format("%.4f", pairTriple.first.third));
				
				pairTriple = Scorer.scoreEventRelationsStructure(resultsFromAllFolds);
				
				LogInfo.logs("Structure Micro precision");
				LogInfo.logs("P : " + String.format("%.4f", pairTriple.first.first));
				LogInfo.logs("R : " + String.format("%.4f", pairTriple.first.second));
				LogInfo.logs("F : " + String.format("%.4f", pairTriple.first.third));
				
				Params eventParam= eventRelationLearner.learn(dataset.examples("train"), eventRelationFeatureFactory);
				
				if(runLocalModel || runBetterBaseline) {
					Utils.writeFile(eventParam, 
							fig.exec.Execution.getActualExecDir() + "/"  + EVENT_RELATION_LOCAL_MODEL_FILE_NAME);
				}
				else if(runGlobalModel) {
					Utils.writeFile(eventParam, 
							fig.exec.Execution.getActualExecDir() + "/"  + EVENT_RELATION_GLOBAL_MODEL_FILE_NAME);
				}
				else if(runLocalBase){
					Utils.writeFile(eventParam, 
							fig.exec.Execution.getActualExecDir() + "/"  + EVENT_RELATION_LOCALBASE_MODEL_FILE_NAME);
				}
				
				/*
				LogInfo.logs("\nFull Macro precision");
				LogInfo.logs("P : " + String.format("%.4f", pairTriple.second.first));
				LogInfo.logs("R : " + String.format("%.4f", pairTriple.second.second));
				LogInfo.logs("F : " + String.format("%.4f", pairTriple.second.third));
				*/

				/*
				System.out.println("Close to one  - " + ILPOptimizer.closeToOne);
				System.out.println("All variables - " + ILPOptimizer.allVariables);
				System.out.println("Percentage    - " + (double)ILPOptimizer.closeToOne / ILPOptimizer.allVariables);
				*/
			}
			//	if(pairTriple.first.third > bestF1) {
			//		bestF1 = pairTriple.first.third;
			//		worstFeature = feature;
			//	}
			//	features.add(feature);
			//}
			
			//LogInfo.logs("Worst feature - "  + worstFeature);
			//LogInfo.logs("Best F1 - "  + bestF1);
			
			//printScores("Dev - Micro", microPrecisionDev, microRecallDev, microF1Dev);
			//printScores("Dev - Macro", macroPrecisionDev, macroRecallDev, macroF1Dev);
			//System.out.println(inferer.totalEvents);
			
			/*
			LogInfo.logs("Maximum number of variables   : " + ILPOptimizer.MaxVariables);
			LogInfo.logs("Maximum number of constraints : " + ILPOptimizer.MaxConstraints);
				
			LogInfo.logs("Previous Event");
			LogInfo.logs("\tActual     " + inferer.prevEvent);
			LogInfo.logs("\tPrediction " + inferer.prevEventPred);
			
			LogInfo.logs("Super Event");
			LogInfo.logs("\tActual     " + inferer.superEvent);
			LogInfo.logs("\tPrediction " + inferer.superEventPred);
			
			LogInfo.logs("Cause Event");
			LogInfo.logs("\tActual     " + inferer.causeEvent);
			LogInfo.logs("\tPrediction " + inferer.causeEventPred);
			
			LogInfo.logs("Degree Distribution");
			LogInfo.logs("\tActual     " + inferer.degreeDistribution);
			LogInfo.logs("\tPrediction " + inferer.degreeDistributionPred);			
			*/
			//LogInfo.logs("Total time:" + EventRelationInferer.runtime);
			//LogInfo.logs("Total time:" + EventRelationInferer.totalRuns);
			//LogInfo.logs("Average time:" + EventRelationInferer.runtime / EventRelationInferer.totalRuns);
			//LogInfo.logs("All times: " + EventRelationInferer.runTimes);
			//Print triples
			/*
			List<String> allRelations = ArgumentRelation.getEventRelations();
			for(String rel1:allRelations) {
				for(String rel2:allRelations) {
					for(String rel3:allRelations) {
						String rel = String.format("%s,%s,%s", rel1, rel2, rel3);
						if(inferer.countGoldTriples.getCount(rel) != 0 || inferer.countPredictedTriples.getCount(rel) !=0)
							LogInfo.logs(String.format("%s, %.0f, %.0f", rel.replace(",", "->"), inferer.countGoldTriples.getCount(rel), inferer.countPredictedTriples.getCount(rel)));
					}
				}
			}
			*/
			/*
			LogInfo.begin_track("Mark relations");
			for(String s:EventRelationFeatureFactory.markWords)
				LogInfo.logs(s);
			LogInfo.end_track();
			
			LogInfo.begin_track("Advmod relations");
			for(String s:EventRelationFeatureFactory.advmodWords)
				LogInfo.logs(s);
			LogInfo.end_track();
			
			LogInfo.begin_track("Event inside PP relations");
			for(String s:EventRelationFeatureFactory.eventInsidePP)
				LogInfo.logs(s);
			LogInfo.end_track();*/
		}
	}
	
	private void runEventPrediction(HashMap<String, String> folders) {
		int NumCrossValidation = 10;
		BioprocessDataset dataset = loadDataSet(folders, false, false);
		CrossValidationSplit split = new CrossValidationSplit(dataset.examples("train"), NumCrossValidation);
		double[] precisionDev = new double[NumCrossValidation], recallDev = new double[NumCrossValidation], f1Dev = new double[NumCrossValidation];

		for(int i = 1; i <= NumCrossValidation; i++) {
			LogInfo.begin_track("Iteration " + i);
			
			Learner entityLearner = new Learner();
			FeatureExtractor entityFeatureFactory = new EntityStandaloneFeatureFactory(useLexicalFeatures);
			Inferer entityInferer = new EntityStandaloneInferer();
			Params entityStandaloneParams = entityLearner.learn(split.GetTrainExamples(i), entityFeatureFactory);
			List<BioDatum> predictedEntities = entityInferer.Infer(split.GetTestExamples(i), entityStandaloneParams, entityFeatureFactory);
			
			Learner eventLearner = new Learner();
			FeatureExtractor eventFeatureFactory = new EventExtendedFeatureFactory(useLexicalFeatures);
			Inferer eventInferer = new EventPredictionInferer(predictedEntities);
			Params eventParam = eventLearner.learn(split.GetTrainExamples(i), eventFeatureFactory);
			List<BioDatum> eventPredicted = eventInferer.Infer(split.GetTestExamples(i), eventParam, eventFeatureFactory);
			
			
			Triple<Double, Double, Double> triple = Scorer.scoreEvents(split.GetTestExamples(i), eventPredicted);
			precisionDev[i-1] = triple.first; recallDev[i-1] = triple.second; f1Dev[i-1] = triple.third;
			LogInfo.end_track();
		}
		printScores("Dev Event trigger prediction", precisionDev, recallDev, f1Dev);
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
			Pair<Triple<Double, Double, Double>, Triple<Double, Double, Double>> triple = opt.optimize(split.GetTrainExamples(i), split.GetTestExamples(i), useLexicalFeatures);
			precisionTrigger[i-1] = triple.first.first; recallTrigger[i-1] = triple.first.second; f1Trigger[i-1] = triple.first.third;
			precisionEntity[i-1] = triple.second.first; recallEntity[i-1] = triple.second.second; f1Entity[i-1] = triple.second.third;
			LogInfo.end_track();
			//break;
		}
		printScores("Dev Trigger", precisionTrigger, recallTrigger, f1Trigger);
		printScores("Dev Entity", precisionEntity, recallEntity, f1Entity);
	}
	
	private BioprocessDataset loadTestDataSet(HashMap<String, String> groups, boolean refreshDataFile) {
		String examplesFileName = "data_test.bpa";
		BioprocessDataset dataset = new BioprocessDataset(groups);

		File f = new File(examplesFileName);
		if(f.exists() && !refreshDataFile) {
			LogInfo.begin_track("Quick data read");
			dataset.allExamples.put("test", Utils.readFile(examplesFileName));
			LogInfo.end_track();
		}
		else {
			dataset.read("test");
			Utils.writeFile(dataset.examples("test"), examplesFileName);
		}
		
		return dataset;
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
	
	private void loadGlobalParameterValues() {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(GLOBAL_PARAM_FILE));
			String line = "";
			while((line = reader.readLine()) != null) {
				String paramName = line.trim().split("\t")[0];
				String paramValue = line.trim().split("\t")[1];
				switch(paramName) {
				case "ConnectedComponent":
					connectedComponent_ = paramValue.equals("true") ? true : false;
					break;
				case "SameEvent":
					sameEvent_ = paramValue.equals("true") ? true : false;
					break;
				case "PreviousEvent":
					previousEvent_ = paramValue.equals("true") ? true : false;
					break;
				case "SameEventContradiction":
					sameEventContradictions_ = paramValue.equals("true") ? true : false;
					break;
				case "Alpha1":
					alpha1_ = Double.parseDouble(paramValue);
					break;
				case "Alpha2":
					alpha2_ = Double.parseDouble(paramValue);
					break;
				case "Alpha3":
					alpha3_ = Double.parseDouble(paramValue);
					break;
				case "Alpha4":
					alpha4_ = Double.parseDouble(paramValue);
					break;
				case "Alpha5":
					alpha5_ = Double.parseDouble(paramValue);
					break;
				case "Alpha6":
					alpha6_ = Double.parseDouble(paramValue);
					break;
				case "Alpha7":
					alpha7_ = Double.parseDouble(paramValue);
					break;
				}
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void computeStats(HashMap<String, String> groups) {
		BioprocessDataset dataset = new BioprocessDataset(groups);
		dataset.read("train");
		dataset.read("test");
		
		System.out.println("Num file - " + BioProcessFormatReader.numFilesRead);
		
		System.out.println("Max sent - " + BioProcessFormatReader.maxSentencesPerProcess);
		System.out.println("Min sent - " + BioProcessFormatReader.minSentencesPerProcess);
		System.out.println("Max toks - " + BioProcessFormatReader.maxTokensPerProcess);
		System.out.println("Min toks - " + BioProcessFormatReader.minTokensPerProcess);
		
		System.out.println("Avg sent - " + ((float)BioProcessFormatReader.numSentences) / BioProcessFormatReader.numFilesRead);
		System.out.println("Avg toks - " + ((float)BioProcessFormatReader.numTokens) / BioProcessFormatReader.numFilesRead);
		int numRelations = 0, numEvents = 0, maxRelations = 0, minRelations = Integer.MAX_VALUE, 
				maxEvents = 0, minEvents = Integer.MAX_VALUE;
		List<String> testAndTrain = new ArrayList<String>();
		testAndTrain.add("test"); testAndTrain.add("train");
		int trainSetPairs = 0, testSetPairs = 0;
		for(String group:testAndTrain) {
			for(Example ex:dataset.examples(group)) {
				int numEventsOnProcess = ex.gold.get(EventMentionsAnnotation.class).size();
				if(numEventsOnProcess > maxEvents) {
					maxEvents = numEventsOnProcess;
					//LogInfo.logs("Updating maxEvents - "+ ex.id);
				}
				if(numEventsOnProcess < minEvents) {
					minEvents = numEventsOnProcess;
					//LogInfo.logs("Updating minEvents - "+ ex.id);
				}
				if(group.equals("test")) {
					testSetPairs += numEventsOnProcess * (numEventsOnProcess-1) / 2;
				}
				else if(group.equals("train")) {
					trainSetPairs += numEventsOnProcess * (numEventsOnProcess-1) / 2;
				}
				numEvents += numEventsOnProcess;
				int numRelationsOnProcess = 0;
				for(EventMention mention:ex.gold.get(EventMentionsAnnotation.class)) {
					for(ArgumentRelation rel: mention.getArguments()) {
						if(ArgumentRelation.getEventRelations().contains(rel.type.toString())) {
							numRelationsOnProcess += 1;
						}
					}
				}
				if(numRelationsOnProcess > maxRelations) {
					maxRelations = numRelationsOnProcess;
					//LogInfo.logs("Updating maxRelations - "+ ex.id);
				}
				
				if(numRelationsOnProcess < minRelations) {
					minRelations = numRelationsOnProcess;
					//LogInfo.logs("Updating minRelations - "+ ex.id);
				}
				
				numRelations += numRelationsOnProcess;
			}
		}
		
		System.out.println("Max even - " + maxEvents);
		System.out.println("Min even - " + minEvents);
		System.out.println("Max rels - " + maxRelations);
		System.out.println("Min rels - " + minRelations);
		
		System.out.println("Avg even - " + ((float)numEvents) / BioProcessFormatReader.numFilesRead);
		System.out.println("Avg rels - " + ((float)numRelations) / BioProcessFormatReader.numFilesRead);
		
		LogInfo.logs("Test set pairs: " + testSetPairs);
		LogInfo.logs("Train set pairs: " + trainSetPairs);
	}
}
