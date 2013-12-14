package edu.stanford.nlp.bioprocess.joint.inference;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;
import edu.stanford.nlp.bioprocess.joint.core.Input;

/**
 * 
 * @author heatherchen1003
 * Constraints that make sure only events have entities, i.e. if t is not classified as an event, it should have no entity.
 */
public class ValidAConstraintGenerator extends ILPConstraintGenerator {

  public ValidAConstraintGenerator() {
    super("~A => ~Exists(B)", false);
  }

  @Override
  public List<ILPConstraint> getILPConstraints(IInstance all,
      InferenceVariableLexManager lexicon) {

    Input input = (Input) all;
    List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
    for (int eventId = 0; eventId < input.getNumberOfTriggers(); eventId++) {
      for (int entityId = 0; entityId < input
          .getNumberOfArgumentCandidates(eventId); entityId++) {
        int[] var = new int[2];
        double[] coef = new double[2];
        var[0] = lexicon.getVariable(Inference.getVariableName(eventId,
            Inference.O_ID, "event"));
        coef[0] = -1;

        var[1] = lexicon.getVariable(Inference.getVariableName(eventId, entityId,
            Inference.NONE_ID, "entity"));
        coef[1] = 1;

        constraints.add(new ILPConstraint(var, coef, 0,
            ILPConstraint.GREATER_THAN));
      }
    }
    return constraints;
  }

  @Override
  public List<ILPConstraint> getViolatedILPConstraints(IInstance arg0,
      IStructure arg1, InferenceVariableLexManager arg2) {
    // This function looks at the structure that is provided as input, and
    // returns only constraints that are violated by it.

    return new ArrayList<ILPConstraint>();
  }

}
