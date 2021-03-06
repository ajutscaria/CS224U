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
import edu.stanford.nlp.bioprocess.joint.core.FeatureExtractor;
import edu.stanford.nlp.bioprocess.joint.core.FeatureVector;
import edu.stanford.nlp.bioprocess.joint.core.Input;
import edu.stanford.nlp.bioprocess.joint.core.Params;
import edu.stanford.nlp.bioprocess.joint.core.Structure;
import edu.stanford.nlp.bioprocess.joint.reader.DatasetUtils;
import fig.basic.LogInfo;

/**
 * 
 * @author heatherchen1003
 * 
 */
public class Inference extends AbstractILPInference<Structure> {
  public static final String[] eventLabels = { DatasetUtils.OTHER_LABEL, DatasetUtils.EVENT_LABEL };
  //public static final String[] entityLabels = { "O", "E" }; 
  public static final String[] entityLabels = {DatasetUtils.NONE_LABEL, DatasetUtils.AGENT, DatasetUtils.THEME,
    DatasetUtils.ORIGIN, DatasetUtils.DESTINATION, DatasetUtils.LOCATION, DatasetUtils.RESULT, DatasetUtils.RAW_MATERIAL,
    DatasetUtils.TIME}; 
  
  public static String[] relationLabels = {DatasetUtils.NONE_LABEL, DatasetUtils.SAME_EVENT, DatasetUtils.PREVIOUS_EVENT,
    DatasetUtils.NEXT_EVENT, DatasetUtils.CAUSE, DatasetUtils.CAUSED, DatasetUtils.ENABLES, DatasetUtils.ENABLED,
    DatasetUtils.SUPER_EVENT,DatasetUtils.SUB_EVENT, DatasetUtils.COTEMPORAL_EVENT};

  // the id for A & B in the above list. This can be done better public static
  public final static int E_ID = 1;
  public final static int O_ID = 0;
  public final static int NONE_ID = 0;

  private List<ILPConstraintGenerator> constraints;
  private Input input;
  private Params params;

  public Inference(Input input, Params params, ILPSolverFactory solverFactory, boolean debug) {
    super(solverFactory, debug);
    this.input = input;
    this.params = params;

    //LogInfo.logs("Start adding constraints for "+input.id);
    constraints = new ArrayList<ILPConstraintGenerator>();
    constraints.add(new UniqueLabelConstraintGenerator());
    constraints.add(new ValidAConstraintGenerator());
    constraints.add(new EntityChildrenConstraintGenerator());
    constraints.add(new RelationEventConstraintGenerator());
    //constraints.add(new PrevRelationConstraintGenerator());
    //constraints.add(new SameRelationConstraintGenerator());
    //constraints.add(new ConnectivityConstraintGenerator());
    //constraints.add(new OverlapConstraintGenerator());
    //LogInfo.logs("finish adding constraints");
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

    //LogInfo.logs("start adding variables");
    addEventEntity(solver, lexicon);
    addEERelation(solver, lexicon);
    //LogInfo.logs("done adding variables");
  }

  private void addEERelation(ILPSolver solver,
      InferenceVariableLexManager lexicon) {
    //LogInfo.logs("start adding event event relation variables");
    for (int Id = 0; Id < input.getNumberOfEERelationCandidates(); Id++) {
      int event1 = input.getEERelationCandidatePair(Id).getSource(); 
      int event2 = input.getEERelationCandidatePair(Id).getTarget(); 
      double score;
      int var;
      String varName;

      // connectivity
      /*score = 0; // ? Yij
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
      */
      
      for (int labelId = 0; labelId < relationLabels.length; labelId++) {
        score = getRelationScore(event1, event2, relationLabels[labelId]);
        var = solver.addBooleanVariable(score);
        varName = getVariableName(event1, event2, labelId, "relation");
        lexicon.addVariable(varName, var);
      }
    }
    //LogInfo.logs("finish adding event entity variables");
  }

  private void addEventEntity(ILPSolver solver,
      InferenceVariableLexManager lexicon) {
    //LogInfo.logs("start adding event entity variables");
    for (int eventId = 0; eventId < input.getNumberOfTriggers(); eventId++) {
      // adding trigger
      for (int labelId = 0; labelId < eventLabels.length; labelId++) {
        double score = getEventScore(eventId, eventLabels[labelId]);
        int var = solver.addBooleanVariable(score);
        String varName = getVariableName(eventId, labelId, "event");
        lexicon.addVariable(varName, var);
      }

      // adding entities for an event
      for (int entityId = 0; entityId < input
          .getNumberOfArgumentCandidates(eventId); entityId++) {
        for (int labelId = 0; labelId < entityLabels.length; labelId++) {
          double score = getEntityScore(eventId, entityId, entityLabels[labelId]);
          int var = solver.addBooleanVariable(score);
          String varName = getVariableName(eventId, entityId, labelId, "entity");
          lexicon.addVariable(varName, var);
        }
      }
    }
    //LogInfo.logs("finish adding event entity variables");
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

  private double getEventScore(int eventId, String label) {
    //LogInfo.logs("getting score for event "+eventId+", label "+label);
    FeatureVector fv = FeatureExtractor.getTriggerLabelFV(input, eventId, label);
    //LogInfo.logs("score for event "+eventId+", label "+label+" = "+fv.dotProduct(params));
    return fv.dotProduct(params);
  }
  
  private double getEntityScore(int eventId, int entityId, String label) {
    //LogInfo.logs("getting score for entity "+entityId+" for event "+eventId+", with label "+label);
    FeatureVector fv = FeatureExtractor.getArgumentLabelFV(input, eventId, entityId, label);
    //LogInfo.logs("score for entity "+entityId+" of event "+eventId+", label "+label+" = "+fv.dotProduct(params));
    return fv.dotProduct(params);
  }
  
  private double getRelationScore(int event1, int event2, String label) {
    //LogInfo.logs("getting score for event-event relation "+event1+", "+event2+", with label "+label);
    FeatureVector fv = FeatureExtractor.getRelationLabelFV(input, event1, event2, label);
    //LogInfo.logs("score for event1 "+event1+", event2 "+event2+", label "+label+" = "+fv.dotProduct(params));
    return fv.dotProduct(params);
  }

  @Override
  protected Structure getOutput(ILPSolver solver,
      InferenceVariableLexManager lexicon) throws Exception {
    
    //LogInfo.logs("Start getting output");
    String[] triggers = new String[input.getNumberOfTriggers()];
    for (int eventId = 0; eventId < input.getNumberOfTriggers(); eventId++) {
      for (int labelId = 0; labelId < eventLabels.length; labelId++) {
        String varName = getVariableName(eventId, labelId, "event");
        int var = lexicon.getVariable(varName);
        if (solver.getBooleanValue(var)){
          triggers[eventId] = eventLabels[labelId];
          break;
        } 
      }
    }

    String[][] arguments = new String[input.getNumberOfTriggers()][];
    for (int eventId = 0; eventId < input.getNumberOfTriggers(); eventId++) {
      arguments[eventId] = new String[input.getNumberOfArgumentCandidates(eventId)];
      for (int entityId = 0; entityId < input
          .getNumberOfArgumentCandidates(eventId); entityId++) {
        for (int labelId = 0; labelId < entityLabels.length; labelId++) {
          String varName = getVariableName(eventId, entityId, labelId, "entity");
          int var = lexicon.getVariable(varName);
          if (solver.getBooleanValue(var)){
            arguments[eventId][entityId] = entityLabels[labelId];
            break;
          } 
        }
      }
    }
    
    String[] relations = new String[input.getNumberOfEERelationCandidates()]; 
    for (int Id = 0; Id < input.getNumberOfEERelationCandidates(); Id++) {
      int event1 = input.getEERelationCandidatePair(Id).getSource(); 
      int event2 = input.getEERelationCandidatePair(Id).getTarget(); 
      for (int labelId = 0; labelId < relationLabels.length; labelId++) {
        String varName = getVariableName(event1, event2, labelId, "relation");
        int var = lexicon.getVariable(varName);
        if (solver.getBooleanValue(var)){
          relations[Id] = relationLabels[labelId];
          break;
        } 
      }
    }
    //LogInfo.logs("Finish getting output");
    //return null;
    return new Structure(input, triggers, arguments, relations);
  }
}
