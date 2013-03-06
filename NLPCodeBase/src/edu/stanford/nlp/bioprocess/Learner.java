package edu.stanford.nlp.bioprocess;

import java.util.List;


/***
 * Class that does the learning
 * @author Aju
 *
 */

public abstract class Learner {
  //Parameters used by the model
  Params parameters;
  //List of examples used to learn the parameters
  List<Example> dataset;
  //Maximum number of iterations to be run
  final static int maxIterations= 100;
  
  /***
   * Constructor to initialize the Learner with a list of training examples.
   * @param ds - List of training examples to learn the parameters from.
   */
  public Learner() {
    parameters = new Params();
  }
  
  /***
   * Method that will learn parameters for the model and return it.
   * @return Parameters learnt.
   */
  public abstract Params learn(List<Example> dataset, FeatureExtractor featureExtractor);
}
