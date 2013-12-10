package edu.stanford.nlp.bioprocess.joint.core;

import java.io.IOException;
import java.util.Random;

import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory;
import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory.SolverType;
import edu.stanford.nlp.bioprocess.joint.inference.Inference;
import edu.stanford.nlp.bioprocess.joint.learn.StructuredPerceptron;
import edu.stanford.nlp.bioprocess.joint.reader.Dataset;
import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;

public class Main implements Runnable {

  public static class Options {
    @Option(gloss = "Dataset dir")
    public String datasetDir;
    @Option(gloss = "Run on dev or test")
    public String runOn;
    @Option(gloss = "Number of folds for cross validation")
    public int numOfFolds = 10;

    // for FeatureExtractor
    @Option(gloss = "use lexical feature or not")
    public boolean useLexicalFeatures = true;
    @Option(gloss = "use baseline feature only or not")
    public boolean useBaselineFeaturesOnly = true;
    @Option(gloss = "run global model or not")
    public boolean runGlobalModel = false;
    @Option(gloss = "window for event-event relation")
    public int window = 10;
  }

  public static Options opts = new Options();

  @Override
  public void run() {
    // TODO Auto-generated method stub
    LogInfo.begin_track("main");
    String trainDirectory = opts.datasetDir + "/train/";
    String testDirectory = opts.datasetDir + "/test/";
    String sampleDirectory = opts.datasetDir + "/sample/";

    Dataset.opts.inPaths.add(Pair.makePair("test", testDirectory));
    Dataset.opts.inPaths.add(Pair.makePair("test", trainDirectory));
    Dataset.opts.inPaths.add(Pair.makePair("test", sampleDirectory));
    Dataset.opts.inFile = "serializeddata";
    Dataset.opts.numOfFolds = opts.numOfFolds;
    Dataset d = new Dataset();
    try {
      d.read();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
      throw new RuntimeException(e);
    } catch (IOException e) {
      e.printStackTrace();
      throw new RuntimeException(e);
    } catch (InterruptedException e) {
      e.printStackTrace();
      throw new RuntimeException(e);
    }
    
    //TODO: Should have a better way to sync?
    FeatureExtractor.opts.runGlobalModel = opts.runGlobalModel;
    FeatureExtractor.opts.useBaselineFeaturesOnly = opts.useBaselineFeaturesOnly;
    FeatureExtractor.opts.useLexicalFeatures = opts.useLexicalFeatures;
    FeatureExtractor.opts.window = opts.window;
    if (opts.runOn.equals("dev")) {
      runJointLearningDev(d);
    } else if (opts.runOn.equals("test")) {
      runJointLearningTest(d);
    }

    LogInfo.end_track();
  }

  private void runJointLearningDev(Dataset d) {//cross-validation
    
    Evaluation eval = new Evaluation(opts.numOfFolds);
    for(int i=0; i<opts.numOfFolds; i++){
      StructuredPerceptron perceptron = new StructuredPerceptron(new Random());
      Params params;
      try {
        params = perceptron.learn(d.getTrainFold(i));
      } catch (Exception e1) {
        e1.printStackTrace();
        throw new RuntimeException(e1);
      }
      for (Pair<Input, Structure> ex : d.getTrainFold(i)) {
        ILPSolverFactory solverFactory = new ILPSolverFactory(
            SolverType.CuttingPlaneGurobi);
        Inference inference = new Inference(ex.first(), params, solverFactory,
            false);
        try {
          Structure predicted = inference.runInference();
          eval.score(ex.second(), predicted, i);
        } catch (Exception e) {
          e.printStackTrace();
          throw new RuntimeException(e);
        }
      }
    } 
    eval.calcScore();
  }
  
  private void runJointLearningTest(Dataset d) {
    StructuredPerceptron perceptron = new StructuredPerceptron(new Random());
    Params params;
    Evaluation eval = new Evaluation(1);
    try {
      params = perceptron.learn(d.examples("train"));
    } catch (Exception e1) {
      e1.printStackTrace();
      throw new RuntimeException(e1);
    }
    for (Pair<Input, Structure> ex : d.examples("test")) {
      ILPSolverFactory solverFactory = new ILPSolverFactory(
          SolverType.CuttingPlaneGurobi);
      Inference inference = new Inference(ex.first(), params, solverFactory,
          false);
      try {
        Structure predicted = inference.runInference();
        eval.score(ex.second(), predicted, 1);
      } catch (Exception e) {
        e.printStackTrace();
        throw new RuntimeException(e);
      }
    }
    eval.calcScore();
    
  }

  /***
   * Entry point to the bio process project.
   * 
   * @param args
   */
  public static void main(String[] args) {
    Execution.run(args, "Main", new Main());
  }

}
