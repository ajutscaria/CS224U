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
import edu.stanford.nlp.bioprocess.ilp.example.ExampleInput;
import edu.stanford.nlp.bioprocess.joint.core.Input;

/**
 * 
 * @author heatherchen1003
 * If a parent entity is classified as an argument for an event t, then its children entities cannot be classified as arguments,
 * and vice versa.
 */
public class EntityChildrenConstraintGenerator extends ILPConstraintGenerator {

  public EntityChildrenConstraintGenerator() {
    super("A => ~Exists(B)", false);
  }

  @Override
  public List<ILPConstraint> getILPConstraints(IInstance x,
      InferenceVariableLexManager lexicon) {

    Input input = (Input) x;
    List<ILPConstraint> constraints = new ArrayList<ILPConstraint>();
    
    // parent is entity -> child is not entity
    for (int eventId = 0; eventId < input.getNumberOfTriggers(); eventId++) {
      Map<Integer, HashSet<Integer>> entityChildren = new HashMap<Integer, HashSet<Integer>>();
      buildEntityChildrenMap(input, entityChildren, eventId);
      //System.out.println("entityChildren size: " + entityChildren.size());
      for (Integer parentId : entityChildren.keySet()) {
        for (Integer childId : entityChildren.get(parentId)) {     
          for (int labelId = 1; labelId < Inference.entityLabels.length; labelId++) {//except NONE
            int[] var = new int[2];
            double[] coef = new double[2];
            var[0] = lexicon.getVariable(Inference.getVariableName(eventId, parentId,
                labelId, "entity"));
            coef[0] = -1;
            var[1] = lexicon.getVariable(Inference.getVariableName(eventId, childId,
                Inference.NONE_ID, "entity"));
            coef[1] = 1;
            constraints.add(new ILPConstraint(var, coef, 0,
                ILPConstraint.GREATER_THAN));
          }
          
        }
  
      }
    }

    return constraints;
  }

  private void buildEntityChildrenMap(Input input,
      Map<Integer, HashSet<Integer>> entityChildren, int eventId) {
    
    for (int entity1 = 0; entity1 < input
        .getNumberOfArgumentCandidates(eventId); entity1++) {
      int entity1Left = input.getArgumentCandidateSpan(eventId, entity1)
          .getSource();
      int entity1Right = input.getArgumentCandidateSpan(eventId, entity1)
          .getTarget();
      for (int entity2 = 0; entity2 < input
          .getNumberOfArgumentCandidates(eventId); entity2++) {
        if (entity1 == entity2)
          continue;
        int entity2Left = input.getArgumentCandidateSpan(eventId, entity2)
            .getSource(); 
        int entity2Right = input.getArgumentCandidateSpan(eventId, entity2)
            .getTarget(); 
        if (entity2Left >= entity1Left && entity2Right <= entity1Right) { //entity2 is a child of entity1
          if (entityChildren.containsKey(entity1)) {
            HashSet<Integer> temp = entityChildren.get(entity1);
            temp.add(entity2);
            entityChildren.put(entity1, temp);
          } else {
            HashSet<Integer> temp = new HashSet<Integer>();
            temp.add(entity2);
            entityChildren.put(entity1, temp);
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
