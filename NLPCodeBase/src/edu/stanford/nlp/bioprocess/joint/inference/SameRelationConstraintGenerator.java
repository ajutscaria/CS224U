package edu.stanford.nlp.bioprocess.joint.inference;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;
import edu.stanford.nlp.bioprocess.BioDatum;
import edu.stanford.nlp.bioprocess.ilp.BioprocessesInput;
import edu.stanford.nlp.bioprocess.ilp.Inference;
import edu.stanford.nlp.bioprocess.joint.core.Input;

/**
 * 
 * @author heatherchen1003
 * Same contradiction. If ti is the same event as tk, their temporal ordering
 * with a third trigger tj may result in a ontradiction.
 */
public class SameRelationConstraintGenerator extends ILPConstraintGenerator {

  public SameRelationConstraintGenerator() {
    super("Yij,* + Yjk,* + Yik,same < = 2", false);
  }

  @Override
  public List<ILPConstraint> getILPConstraints(IInstance all,
      InferenceVariableLexManager lexicon) {

    Input input = (Input) all;
    List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
    int slotlength = Inference.relationLabels.length;
    for (int i = 0; i < input.getNumberOfTriggers() - 2; i++) {
      for (int j = i + 1; j < input.getNumberOfTriggers() - 1; j++) {
        for (int k = j + 1; k < input.getNumberOfTriggers(); k++) {
          if (!input.isRelationCandidate(i, j)
              || !input.isRelationCandidate(j, k)
              || !input.isRelationCandidate(i, k))
            continue;
          int labelId1 = 1; // SAME
          for (int labelId2 = 2; labelId2 < slotlength; labelId2++) { // except "NONE" or "SAME"
            for (int labelId3 = 2; labelId3 < slotlength; labelId3++) {
              int[] var = new int[3];
              double[] coef = new double[3];
              var[0] = lexicon.getVariable(Inference.getVariableName(i, k,
                  labelId1, "relation"));
              coef[0] = 1;
              var[1] = lexicon.getVariable(Inference.getVariableName(i, j,
                  labelId2, "relation"));
              coef[1] = 1;
              var[2] = lexicon.getVariable(Inference.getVariableName(j, k,
                  labelId3, "relation"));
              coef[2] = 1;
              constraints.add(new ILPConstraint(var, coef, 2,
                  ILPConstraint.LESS_THAN));
            }
          }

          for (int labelId2 = 2; labelId2 < slotlength; labelId2++) { // except "NONE" or "SAME"
            for (int labelId3 = 2; labelId3 < slotlength; labelId3++) {
              int[] var = new int[3];
              double[] coef = new double[3];
              var[0] = lexicon.getVariable(Inference.getVariableName(j, k,
                  labelId1, "relation"));
              coef[0] = 1;
              var[1] = lexicon.getVariable(Inference.getVariableName(i, j,
                  labelId2, "relation"));
              coef[1] = 1;
              var[2] = lexicon.getVariable(Inference.getVariableName(i, k,
                  labelId3, "relation"));
              coef[2] = 1;
              constraints.add(new ILPConstraint(var, coef, 2,
                  ILPConstraint.LESS_THAN));
            }
          }

          for (int labelId2 = 2; labelId2 < slotlength; labelId2++) { // except "NONE" or "SAME"
            for (int labelId3 = 2; labelId3 < slotlength; labelId3++) {
              int[] var = new int[3];
              double[] coef = new double[3];
              var[0] = lexicon.getVariable(Inference.getVariableName(i, j,
                  labelId1, "relation"));
              coef[0] = 1;
              var[1] = lexicon.getVariable(Inference.getVariableName(i, k,
                  labelId2, "relation"));
              coef[1] = 1;
              var[2] = lexicon.getVariable(Inference.getVariableName(j, k,
                  labelId3, "relation"));
              coef[2] = 1;
              constraints.add(new ILPConstraint(var, coef, 2,
                  ILPConstraint.LESS_THAN));
            }
          }
        }
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
