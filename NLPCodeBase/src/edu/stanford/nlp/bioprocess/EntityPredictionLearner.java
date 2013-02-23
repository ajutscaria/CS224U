package edu.stanford.nlp.bioprocess;

import java.util.List;


/***
 * Class that does the learning
 * @author Aju
 *
 */

public class EntityPredictionLearner extends Learner{
  public EntityPredictionLearner(List<Example> ds) {
		super(ds);
  }

//Parameters used by the model
  Params parameters;
  //List of examples used to learn the parameters
  List<Example> dataset;
  //Maximum number of iterations to be run
  final static int maxIterations = 100;
  
  /***
   * Method that will learn parameters for the model and return it.
   * @return Parameters learnt.
   */
  public Params learn() {
    return parameters;
  }
}
