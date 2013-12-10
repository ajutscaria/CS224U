package edu.stanford.nlp.bioprocess.joint.core;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory;
import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory.SolverType;
import edu.stanford.nlp.bioprocess.EntityFeatureFactory;
import edu.stanford.nlp.bioprocess.EntityPredictionInferer;
import edu.stanford.nlp.bioprocess.EventExtendedFeatureFactory;
import edu.stanford.nlp.bioprocess.EventPredictionInferer;
import edu.stanford.nlp.bioprocess.Learner;
import edu.stanford.nlp.bioprocess.Scorer;
import edu.stanford.nlp.bioprocess.Utils;
import edu.stanford.nlp.bioprocess.joint.inference.Inference;
import edu.stanford.nlp.bioprocess.joint.reader.Dataset;
import edu.stanford.nlp.bioprocess.joint.reader.Dataset.Options;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.Pair;
import fig.exec.Execution;

public class Main implements Runnable{

  public static class Options {
    @Option(gloss = "Dataset dir")
    public String datasetDir;
    
    //for FeatureExtractor
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
    String trainDirectory = opts.datasetDir+"/train/";
    String testDirectory = opts.datasetDir+"/test/";
    String sampleDirectory = opts.datasetDir+"/sample/";
    
    Dataset.opts.inPaths.add(Pair.makePair("test",testDirectory));
    Dataset.opts.inPaths.add(Pair.makePair("test",trainDirectory));
    Dataset.opts.inPaths.add(Pair.makePair("test", sampleDirectory));
    Dataset.opts.inFile = "serializeddata";
    Dataset d = new Dataset();
    try {
      d.read();
    } catch (ClassNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
      throw new RuntimeException(e);
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
      throw new RuntimeException(e);
    } catch (InterruptedException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
      throw new RuntimeException(e);
    }
    
    Params params = new Params(); // = new StructurePerceptron(d.examples("train"));
    for(Pair<Input,Structure> ex: d.examples("test")){
      ILPSolverFactory solverFactory = new ILPSolverFactory(SolverType.CuttingPlaneGurobi);
      Inference inference = new Inference(ex.getFirst(), params, solverFactory, false);
      try {
          Structure predicted = inference.runInference();
          Evaluation.score(ex.getSecond(), predicted);
      } catch (Exception e) {
          e.printStackTrace();
          throw new RuntimeException(e);
      }
      
    }
    
    LogInfo.end_track();
  }
  

  /***
   * Entry point to the bio process project. 
   * @param args
   */  
  public static void main(String[] args) {
    Execution.run(args,
            "Main", new Main());
  }

}
