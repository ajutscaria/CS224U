package edu.stanford.nlp.bioprocess.joint.inference;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.infer.ilp.AbstractILPInference;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.ILPSolver;
import edu.illinois.cs.cogcomp.infer.ilp.ILPSolverFactory;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;
import edu.stanford.nlp.bioprocess.joint.core.Input;
import edu.stanford.nlp.bioprocess.joint.core.Structure;

/**
 * 
 * @author heatherchen1003
 * 
 */
public class Inference extends AbstractILPInference<Structure> {
  public static final String[] eventLabels = { "E", "O" };
  public static final String[] entityLabels = { "E", "O" }; // Can chagne into
                                                            // multi-class
  public static String[] relationLabels = {};

  // the id for A & B in the above list. This can be done better public static
  public final static int E_ID = 0;
  public final static int O_ID = 1;

  private List<ILPConstraintGenerator> constraints;
  private Input input;

  public Inference(Input input, ILPSolverFactory solverFactory, boolean debug) {
    super(solverFactory, debug);
    this.input = input;

    constraints = new ArrayList<ILPConstraintGenerator>();
    constraints.add(new UniqueLabelConstraintGenerator());
    constraints.add(new ValidAConstraintGenerator());
    constraints.add(new EntityChildrenConstraintGenerator());
    constraints.add(new RelationEventConstraintGenerator());
    constraints.add(new PrevRelationConstraintGenerator());
    constraints.add(new SameRelationConstraintGenerator());
    constraints.add(new ConnectivityConstraintGenerator());
    constraints.add(new OverlapConstraintGenerator());
  }

  @Override
  protected void addConstraints(ILPSolver solver,
      InferenceVariableLexManager lexicon) {

    for (ILPConstraintGenerator c : constraints) {
      for (ILPConstraint constraint : c.getILPConstraints(input, lexicon))
        this.addConstraint(solver, constraint);
    }
  }

  @Override
  protected void addVariables(ILPSolver solver,
      InferenceVariableLexManager lexicon) {

    System.out.println("start adding variables");
    for (int eventId = 0; eventId < input.getNumberOfTriggers(); eventId++) {
      // adding trigger
      double score = 0;
      int var = solver.addBooleanVariable(score);
      String varName = getVariableName(eventId, E_ID, "event"); // 0: E
      lexicon.addVariable(varName, var);

      score = 0;
      var = solver.addBooleanVariable(score);
      varName = getVariableName(eventId, O_ID, "event"); // 0: E, 1: O
      lexicon.addVariable(varName, var);

      // adding entities for an event
      for (int entityId = 0; entityId < input
          .getNumberOfArgumentCandidates(eventId); entityId++) {
        for (int labelId = 0; labelId < entityLabels.length; labelId++) {
          score = 0;
          var = solver.addBooleanVariable(score);
          varName = getVariableName(entityId, labelId, "entity");
          lexicon.addVariable(varName, var);
        }
      }
    }

    for (int Id = 0; Id < input.getNumberOfEERelationCandidates(); Id++) {
      int event1 = input.getEERelationCandidatePair(Id).getSource(); // ?
      int event2 = input.getEERelationCandidatePair(Id).getTarget(); // ?
      double score;
      int var;
      String varName;

      // connectivity
      score = 0; // ? Yij
      var = solver.addBooleanVariable(score);
      varName = getVariableName(event1, event2, "edge", "connectivity");
      lexicon.addVariable(varName, var);
      // System.out.println(varName);
      score = 0;// ? Zij
      var = solver.addBooleanVariable(score);
      varName = getVariableName(event1, event2, "aux", "connectivity");
      // System.out.println(varName);
      lexicon.addVariable(varName, var);

      score = 0;// ? PHIij
      // var = solver.addRealVariable(score);
      var = solver.addIntegerVariable(score);
      varName = getVariableName(event1, event2, "flow", "connectivity");
      lexicon.addVariable(varName, var);
      // System.out.println(varName);

      score = 0;// ? Zji
      var = solver.addBooleanVariable(score);
      varName = getVariableName(event2, event1, "aux", "connectivity");
      lexicon.addVariable(varName, var);
      // System.out.println(varName);

      score = 0;// ? PHIji
      // var = solver.addRealVariable(score);
      var = solver.addIntegerVariable(score);
      varName = getVariableName(event2, event1, "flow", "connectivity");
      lexicon.addVariable(varName, var);

      for (int labelId = 0; labelId < relationLabels.length; labelId++) {
        score = 0;
        var = solver.addBooleanVariable(score);
        varName = getVariableName(event1, event2, labelId, "relation");
        lexicon.addVariable(varName, var);
      }
    }
    System.out.println("done adding variables");
  }

  public static String getVariableName(int slotId, int labelId) {
    return "slot" + slotId + "-" + labelId;
  }

  public static String getVariableName(int eventId, int labelId, String type) {
    return type + eventId + ",label" + labelId;
  }

  public static String getVariableName(int event1, int event2, String type,
      String special) {
    return type + event1 + "," + event2;
  }

  public static String getVariableName(int event1, int event2, int labelId,
      String type) {
    return type + event1 + " " + event2 + ",label" + labelId;
  }

  private double getLabelScore(int slotId, int labelId) {
    // for now, some random scores
    return (new Random()).nextDouble();
  }

  @Override
  protected Structure getOutput(ILPSolver solver,
      InferenceVariableLexManager lexicon) throws Exception {

    boolean[] triggers = new boolean[input.getNumberOfTriggers()];
    for (int eventId = 0; eventId < input.getNumberOfTriggers(); eventId++) {
      for (int labelId = 0; labelId < eventLabels.length; labelId++) {
        String varName = getVariableName(eventId, labelId, "event");
        int var = lexicon.getVariable(varName);

        if (solver.getBooleanValue(var) && labelId == E_ID) {
          triggers[eventId] = true;
          break;
        } else {
          triggers[eventId] = false;
        }
      }
    }

    String[][] arguments = new String[input.getNumberOfTriggers()][];
    String[][] relations; // ?

    return null;// new ExampleStructure(input, labels);
  }
}
