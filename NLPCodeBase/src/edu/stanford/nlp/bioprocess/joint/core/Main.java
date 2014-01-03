package edu.stanford.nlp.bioprocess.joint.core;

import java.io.IOException;
import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory;
import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory.SolverType;
import edu.stanford.nlp.bioprocess.joint.inference.Inference;
import edu.stanford.nlp.bioprocess.joint.learn.StructuredPerceptron;
import edu.stanford.nlp.bioprocess.joint.reader.Dataset;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;

public class Main implements Runnable {

  public static class Options {
    @Option(gloss = "Dataset dir")
    public String datasetdir;
    @Option(gloss = "Run on dev or test")
    public String runon;

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

  @Option(gloss = "Dataset dir")
  public String datasetDir;
  @Option(gloss = "Run mode (dev or test currently)")
  public String mode;

  @Override
  public void run() {
    // TODO Auto-generated method stub
    LogInfo.begin_track("main");
    String trainDirectory = datasetDir + "/train/";
    String testDirectory = datasetDir + "/test/";

    Dataset.opts.inPaths.add(Pair.makePair("test", testDirectory));
    Dataset.opts.inPaths.add(Pair.makePair("train", trainDirectory));
    // Dataset.opts.inPaths.add(Pair.makePair("sample", sampleDirectory));
    Dataset.opts.inFile = "serializeddata";
    Dataset d = new Dataset();
    try {
      d.read();
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    } catch (IOException e) {
      throw new RuntimeException(e);
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }

    // TODO: Should have a better way to sync?
    /*
     * FeatureExtractor.opts.runGlobalModel = opts.runGlobalModel;
     * FeatureExtractor.opts.useBaselineFeaturesOnly =
     * opts.useBaselineFeaturesOnly; FeatureExtractor.opts.useLexicalFeatures =
     * opts.useLexicalFeatures; FeatureExtractor.opts.window = opts.window;
     */
    try {
      if (mode.equals("dev")) {
        runJointLearningDev(d);
      } else if (mode.equals("test")) {
        //runJointLearningTest(d);
        testInference(d);    
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    LogInfo.end_track();
  }

  private void runJointLearningDev(Dataset d) {// cross-validation

    Evaluation eval = new Evaluation(Dataset.opts.numOfFolds);
    for (int i = 0; i < Dataset.opts.numOfFolds; i++) {
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

  private void testInference(Dataset d) throws Exception {
    StructuredPerceptron perceptron = new StructuredPerceptron(new Random(1));
    //Params params = new Params();
    Params params = perceptron.learn(d.examples("train"));
    Evaluation eval = new Evaluation(1);
    List<Pair<Input, Structure>> dataset = d.examples("train");

    Input input = dataset.get(0).first;
    Structure structure = dataset.get(0).second;

    //LogInfo.logs(input.toString());
    //LogInfo.logs(structure);

    ILPSolverFactory solverFactory = new ILPSolverFactory(
        SolverType.CuttingPlaneGurobi);
    Inference inference = new Inference(input, params, solverFactory, false);

    Structure prediction = inference.runInference();
    eval.score(structure, prediction, 0);
    eval.calcScore();
    //LogInfo.logs(prediction);
  }

  private void runJointLearningTest(Dataset d) throws Exception {
    StructuredPerceptron perceptron = new StructuredPerceptron(new Random(1));
    Evaluation eval = new Evaluation(1); // we are guaranteed that test is always
                                         // exactly one fold
    Params params = perceptron.learn(d.examples("train"));

    for (Pair<Input, Structure> ex : d.examples("test")) {
      ILPSolverFactory solverFactory = new ILPSolverFactory(
          SolverType.CuttingPlaneGurobi);
      Inference inference = new Inference(ex.first(), params, solverFactory,
          false);

      Structure predicted = inference.runInference();
      eval.score(ex.second(), predicted, 0);

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
