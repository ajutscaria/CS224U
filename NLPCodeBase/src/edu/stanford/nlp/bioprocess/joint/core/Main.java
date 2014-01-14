package edu.stanford.nlp.bioprocess.joint.core;

import java.io.IOException;
import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory;
import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory.SolverType;
import edu.stanford.nlp.bioprocess.joint.inference.Inference;
import edu.stanford.nlp.bioprocess.joint.learn.StructuredPerceptron;
import edu.stanford.nlp.bioprocess.joint.learn.StructuredPerceptron.EpochReporter;
import edu.stanford.nlp.bioprocess.joint.learn.StructuredPerceptron.ExampleReporter;
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

  private void runJointLearningDev(final Dataset d) {// cross-validation

    final Evaluation allFolds = new Evaluation(Dataset.opts.numOfFolds);
    // for (int i = 0; i < Dataset.opts.numOfFolds; i++) {
    for (int i = 0; i < Dataset.opts.numOfFolds; i++) {
      final int fold = i;
      LogInfo.begin_track("Fold " + (i+1));
      ExampleReporter exampleReporter = new ExampleReporter() {
        Evaluation e;// = new Evaluation(1);
        
        @Override
        public void report(Input example, Structure gold, Structure predicted) {
          // TODO Auto-generated method stub  
          LogInfo.begin_track("Process "+example.id);
          e.score(gold, predicted, 0); 
          LogInfo.end_track();
        }
        
        public void clear(){
          e = new Evaluation(1);
        }
      };
      EpochReporter epochReporter = new EpochReporter() {

        Evaluation eval;// = new Evaluation(1);
        @Override
        public void report(int epochId, Params w, Params avg,
            ILPSolverFactory solverFactory) {
          // TODO Auto-generated method stub
          LogInfo.begin_track("Prediction on development set after Epoch "+epochId);
          for(Pair<Input, Structure> ex : d.getDevFold(fold)) { 
            Inference inference = new Inference(ex.first(), w, solverFactory,false); 
            try { 
              Structure predicted = inference.runInference();
              LogInfo.begin_track("Prediction for example "+ex.first.id);
              eval.score(ex.second(), predicted, 0); 
              if(epochId == StructuredPerceptron.opts.numEpochs-1){
                LogInfo.begin_track("Gold Structure:");
                LogInfo.logs(ex.second().toString());
                LogInfo.end_track();
                LogInfo.begin_track("Predicted Structure:");
                LogInfo.logs(predicted.toString());
                LogInfo.end_track();
              }
              LogInfo.end_track(); 
            } catch(Exception e) { 
              e.printStackTrace(); 
              throw new RuntimeException(e); 
            } 
          }
          LogInfo.end_track();
          if(epochId == StructuredPerceptron.opts.numEpochs-1){ // last Epoch for this fold
            allFolds.setEventMeasure(fold, eval.getEventMeasure(0));
            allFolds.setEntityMeasure(fold, eval.getEntityMeasure(0));
            allFolds.setEERelationMeasure(fold, eval.getEERelationMeasure(0));
          }
        }

        @Override
        public void recordExample(Input example, Structure gold,
            Structure predicted) {
          // TODO Auto-generated method stub

        }

        @Override
        public void clear() {
          // TODO Auto-generated method stub
          eval = new Evaluation(1);
        }
      };

      StructuredPerceptron perceptron = new StructuredPerceptron(new Random(1),
          exampleReporter, epochReporter);
      Params params = new Params(); 
      
      try { 
        params = perceptron.learn(d.getTrainFold(i)); 
      } catch (Exception e1) { 
        e1.printStackTrace(); 
        throw new RuntimeException(e1); 
      } 
      
      LogInfo.end_track();
    }
    allFolds.calcScore();
  }

  private void testInference(Dataset d) throws Exception {
    ExampleReporter exampleReporter = new ExampleReporter() {

      @Override
      public void report(Input example, Structure gold, Structure predicted) {
        // TODO Auto-generated method stub

      }

      @Override
      public void clear() {
        // TODO Auto-generated method stub
        
      }
    };
    EpochReporter epochReporter = new EpochReporter() {

      @Override
      public void report(int epochId, Params w, Params avg,
          ILPSolverFactory solverFactory) {
        // TODO Auto-generated method stub
        
      }

      @Override
      public void recordExample(Input example, Structure gold,
          Structure predicted) {
        // TODO Auto-generated method stub

      }

      @Override
      public void clear() {
        // TODO Auto-generated method stub

      }
    };

    //StructuredPerceptron perceptron = new StructuredPerceptron(new Random(1), exampleReporter, epochReporter);
    Params params = new Params();
    //Params params = perceptron.learn(d.examples("train"));
    Evaluation eval = new Evaluation(1);
    List<Pair<Input, Structure>> dataset = d.examples("test");

    Input input = dataset.get(0).first;
    Structure structure = dataset.get(0).second;

    // LogInfo.logs(input.toString());
    // LogInfo.logs(structure);

    ILPSolverFactory solverFactory = new ILPSolverFactory(
        SolverType.CuttingPlaneGurobi);
    Inference inference = new Inference(input, params, solverFactory, false);

    Structure prediction = inference.runInference();
    eval.score(structure, prediction, 0);
    eval.calcScore();
    // LogInfo.logs(prediction);
  }

  private void runJointLearningTest(Dataset d) throws Exception {
    ExampleReporter exampleReporter = new ExampleReporter() {

      @Override
      public void report(Input example, Structure gold, Structure predicted) {
        // TODO Auto-generated method stub

      }

      @Override
      public void clear() {
        // TODO Auto-generated method stub
        
      }
    };
    EpochReporter epochReporter = new EpochReporter() {

      @Override
      public void report(int epochId, Params w, Params avg,
          ILPSolverFactory solverFactory) {
        // TODO Auto-generated method stub
        
      }

      @Override
      public void recordExample(Input example, Structure gold,
          Structure predicted) {
        // TODO Auto-generated method stub

      }

      @Override
      public void clear() {
        // TODO Auto-generated method stub

      }
    };

    StructuredPerceptron perceptron = new StructuredPerceptron(new Random(1), exampleReporter, epochReporter);
    Evaluation eval = new Evaluation(1); // we are guaranteed that test is
                                         // always
                                         // exactly one fold
    Params params = perceptron.learn(d.examples("train"));

    LogInfo.logs("Start predicting on the test set");
    for (Pair<Input, Structure> ex : d.examples("test")) {
      ILPSolverFactory solverFactory = new ILPSolverFactory(
          SolverType.CuttingPlaneGurobi);
      Inference inference = new Inference(ex.first(), params, solverFactory,
          false);

      Structure predicted = inference.runInference();
      LogInfo.begin_track("Predicting " + ex.first().id);
      eval.score(ex.second(), predicted, 0);
      LogInfo.end_track();
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
