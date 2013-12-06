package edu.stanford.nlp.bioprocess.joint.inference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;
import edu.stanford.nlp.bioprocess.BioDatum;
import edu.stanford.nlp.bioprocess.joint.core.Input;
import fig.basic.LogInfo;

/**
 * 
 * @author heatherchen1003
 * Constraints that ensure the event graph is connected (each event belongs to at least one relation).
 */
public class ConnectivityConstraintGenerator extends ILPConstraintGenerator {

  public ConnectivityConstraintGenerator() {
    super("Zij, Yij, PHIij", false);
  }

  @Override
  public List<ILPConstraint> getILPConstraints(IInstance all,
      InferenceVariableLexManager lexicon) {

    Input input = (Input) all;
    List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
    bindingRelationEdge(lexicon, input, constraints);

    // equation 2
    auxiliaryTying(lexicon, input, constraints);

    int size = input.getNumberOfTriggers() - 1;
    int root = 0;  // choose the smallest-index event in all the relations as root
    for (int i = 0; i < input.getNumberOfTriggers(); i++) {
      for (int j = i+1; j < input.getNumberOfTriggers(); j++) {
        if (input.isRelationCandidate(i, j)){
          root = i;
          break;
        }
      }
    }
   
    // equation 3
    addOneParentConstraint(lexicon, input, constraints, size, root);

    // equation 4
    addRootFlowConstraint(lexicon, constraints, input, size, root);

    // equation 5
    addFlowConsistencyConstraints(lexicon, constraints, input, size, root);

    // equation 6
    // System.out.println("equation 6");
    addFlowEdgeVariableConstraints(lexicon, constraints, input);

    return constraints;
  }

  private void addOneParentConstraint(InferenceVariableLexManager lexicon,
      Input input, List<ILPConstraint> constraints, int size, int root) {
    for (int j = 0; j < input.getNumberOfTriggers(); j++) {
      int[] var = new int[size];
      double[] coef = new double[size];
      int counter = 0;
      int event2 = j;

      for (int i = 0; i < input.getNumberOfTriggers(); i++) {
        int event1 = i;
        if (i == j || !input.isRelationCandidate(i, j))
          continue;
        var[counter] = lexicon.getVariable(Inference.getVariableName(event1,
            event2, "aux", "connectivity"));
        coef[counter] = 1;
        counter++;
      }
      if (j == root) {// root
        constraints.add(new ILPConstraint(var, coef, 0, ILPConstraint.EQUAL));
      } else {
        constraints.add(new ILPConstraint(var, coef, 1, ILPConstraint.EQUAL));
      }
    }
  }

  private void auxiliaryTying(InferenceVariableLexManager lexicon, Input input,
      List<ILPConstraint> constraints) {
    for (int Id = 0; Id < input.getNumberOfEERelationCandidates(); Id++) {
      int event1 = input.getEERelationCandidatePair(Id).getSource(); // ?
      int event2 = input.getEERelationCandidatePair(Id).getTarget(); // ?

      // equation 2
      int[] var = new int[2];
      double[] coef = new double[2];
      // Zij <= Yij
      var[0] = lexicon.getVariable(Inference.getVariableName(event1, event2,
          "edge", "connectivity")); // Yij
      coef[0] = -1;
      var[1] = lexicon.getVariable(Inference.getVariableName(event1, event2,
          "aux", "connectivity")); // Zij
      coef[1] = 1;
      constraints.add(new ILPConstraint(var, coef, 0, ILPConstraint.LESS_THAN));
      // Zji <= Yij
      var = new int[2];
      coef = new double[2];
      var[0] = lexicon.getVariable(Inference.getVariableName(event1, event2,
          "edge", "connectivity")); // Yij
      coef[0] = -1;
      var[1] = lexicon.getVariable(Inference.getVariableName(event2, event1,
          "aux", "connectivity")); // Zji
      coef[1] = 1;
      constraints.add(new ILPConstraint(var, coef, 0, ILPConstraint.LESS_THAN));
    }
  }

  private void bindingRelationEdge(InferenceVariableLexManager lexicon,
      Input input, List<ILPConstraint> constraints) {
    int slotlength = Inference.relationLabels.length;

    for (int Id = 0; Id < input.getNumberOfEERelationCandidates(); Id++) {
      int event1 = input.getEERelationCandidatePair(Id).getSource(); // ?
      int event2 = input.getEERelationCandidatePair(Id).getTarget(); // ?

      // Yij = SUM(Yijr), r != NONE
      int[] vars = new int[slotlength];
      double[] coefficients = new double[slotlength];
      for (int labelId = 1; labelId < slotlength; labelId++) {
        vars[labelId] = lexicon.getVariable(Inference.getVariableName(event1,
            event2, labelId, "relation"));
        coefficients[labelId] = 1.0;
      }
      vars[0] = lexicon.getVariable(Inference.getVariableName(event1, event2,
          "edge", "connectivity")); // Yij
      coefficients[0] = -1;
      constraints.add(new ILPConstraint(vars, coefficients, 0,
          ILPConstraint.EQUAL));
    }
  }

  private void addRootFlowConstraint(InferenceVariableLexManager lexicon,
      List<ILPConstraint> constraints, Input input, int size, int root) {
    int counter = 0;
    int[] var = new int[size];
    double[] coef = new double[size];
    for (int i = 0; i < input.getNumberOfTriggers(); i++) {
      int event2 = i;
      if(i==root || !input.isRelationCandidate(root, event2))continue;
      var[counter] = lexicon.getVariable(Inference.getVariableName(root,
          event2, "flow", "connectivity"));
      coef[counter] = 1;
      counter++;
    }
    constraints.add(new ILPConstraint(var, coef, counter, ILPConstraint.EQUAL));
  }

  private void addFlowConsistencyConstraints(
      InferenceVariableLexManager lexicon, List<ILPConstraint> constraints,
      Input input, int size, int root) {
    int counter;
    int[] var;
    double[] coef;
    int doublesize = 2 * size;
    for (int j = 0; j < input.getNumberOfTriggers(); j++) {
      if(j==root)continue;
      int eventj = j;
      var = new int[doublesize];
      coef = new double[doublesize];
      counter = 0;
      for (int i = 0; i < input.getNumberOfTriggers(); i++) {
        if (i == j || !input.isRelationCandidate(i, j))
          continue;
        int eventi = i;
        var[counter] = lexicon.getVariable(Inference.getVariableName(eventi,
            eventj, "flow", "connectivity"));
        coef[counter] = 1;
        counter++;
      }
      for (int k = 0; k < input.getNumberOfTriggers(); k++) {
        if (j == k || !input.isRelationCandidate(j, k))
          continue;
        int eventk = k;
        var[counter] = lexicon.getVariable(Inference.getVariableName(eventj,
            eventk, "flow", "connectivity"));
        coef[counter] = -1;
        counter++;
      }
      constraints.add(new ILPConstraint(var, coef, 1, ILPConstraint.EQUAL));
    }
  }

  private void addFlowEdgeVariableConstraints(
      InferenceVariableLexManager lexicon, List<ILPConstraint> constraints,
      Input input) {
    int n = input.getNumberOfTriggers();
    // LogInfo.logs("Events=%s", events);
    for (int i = 0; i < n; i++) {
      int eventi = i;

      for (int j = 0; j < n; j++) {
        if (i == j || !input.isRelationCandidate(i, j))
          continue;
        int eventj = j;
        int[] var = new int[2];
        double[] coef = new double[2];
        String phiij = Inference.getVariableName(eventi, eventj, "flow",
            "connectivity");
        var[0] = lexicon.getVariable(phiij); // PHIij
        coef[0] = 1;

        String zij = Inference.getVariableName(eventi, eventj, "aux",
            "connectivity");
        var[1] = lexicon.getVariable(zij); // Zij
        coef[1] = -n;

        ILPConstraint constraint = new ILPConstraint(var, coef, 0,
            ILPConstraint.LESS_THAN);

        // LogInfo.logs("Constraint: %s", constraint.toString());
        constraints.add(constraint);

        // flow is positive
        int[] flowvar = new int[1];
        double[] flowcoef = new double[1];
        flowvar[0] = lexicon.getVariable(phiij);
        flowcoef[0] = 1;
        ILPConstraint posflowconstraint = new ILPConstraint(flowvar, flowcoef,
            0, ILPConstraint.GREATER_THAN);

        // LogInfo.logs("Constraint: %s", constraint.toString());
        constraints.add(posflowconstraint);
      }
    }
  }

  @Override
  public List<ILPConstraint> getViolatedILPConstraints(IInstance arg0,
      IStructure arg1, InferenceVariableLexManager arg2) {
    // This function looks at the structure that is provided as input, and
    // returns only constraints that are violated by it.

    return new ArrayList<ILPConstraint>();
  }

}
