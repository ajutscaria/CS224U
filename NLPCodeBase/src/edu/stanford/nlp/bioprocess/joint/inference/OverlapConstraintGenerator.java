package edu.stanford.nlp.bioprocess.joint.inference;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraint;
import edu.illinois.cs.cogcomp.infer.ilp.ILPConstraintGenerator;
import edu.illinois.cs.cogcomp.infer.ilp.InferenceVariableLexManager;
import edu.stanford.nlp.bioprocess.ilp.BioprocessesInput;
import edu.stanford.nlp.bioprocess.ilp.Inference;
import edu.stanford.nlp.bioprocess.ilp.example.ExampleInput;
import edu.stanford.nlp.bioprocess.joint.core.Input;
import edu.stanford.nlp.util.IntPair;

/**
 * 
 * @author heatherchen1003
 * If trigger t is an event, candidate arguments that overlap with t should not be 
 * classified as entities, and vice versa.
 */
public class OverlapConstraintGenerator extends ILPConstraintGenerator {

  public OverlapConstraintGenerator() {
    super("A => Exists(B)", false);
  }

  @Override
  public List<ILPConstraint> getILPConstraints(IInstance x,
      InferenceVariableLexManager lexicon) {

    Input input = (Input)x;
    List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
    Map<Integer, HashSet<IntPair>> entityOverlapEvent = new HashMap<Integer, HashSet<IntPair>>();
    buildEntityOverlapEventMap(input, entityOverlapEvent);
    
    //event -> overlapped argument candidates should not be entities
    for (Integer parentId : entityOverlapEvent.keySet()) {
      for (IntPair childId : entityOverlapEvent.get(parentId)) {
        int[] var = new int[2];
        double[] coef = new double[2];

        var[0] = lexicon.getVariable(Inference.getVariableName(parentId,
            Inference.E_ID, "event"));
        coef[0] = -1;
        var[1] = lexicon.getVariable(Inference.getVariableName(childId.getSource(), childId.getTarget(),
            Inference.O_ID, "entity"));
        coef[1] = 1;
        constraints.add(new ILPConstraint(var, coef, 0,
            ILPConstraint.GREATER_THAN));
      }

    } 
    
    return constraints;
  }
  
  private void buildEntityOverlapEventMap(Input input,
      Map<Integer, HashSet<IntPair>> entityOverlapEvent) {  
    for (int event = 0; event < input.getNumberOfTriggers(); event++) {
      int tokenId = input.getTriggerTokenId(event);
      for (int eventId = 0; eventId < input.getNumberOfTriggers(); eventId++) {
        for (int entity1 = 0; entity1 < input
            .getNumberOfArgumentCandidates(eventId); entity1++){
          int entity1Left = input.getArgumentCandidateSpan(eventId, entity1)
              .getSource();
          int entity1Right = input.getArgumentCandidateSpan(eventId, entity1)
              .getTarget();
          if (entity1Left <= tokenId && entity1Right >= tokenId) { //entity1 includes the event
            if (entityOverlapEvent.containsKey(event)) {
              HashSet<IntPair> temp = entityOverlapEvent.get(event);
              temp.add(new IntPair(eventId, entity1));
              entityOverlapEvent.put(event, temp);
            } else {
              HashSet<IntPair> temp = new HashSet<IntPair>();
              temp.add(new IntPair(eventId, entity1));
              entityOverlapEvent.put(event, temp);
            }
          }
        } 
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
