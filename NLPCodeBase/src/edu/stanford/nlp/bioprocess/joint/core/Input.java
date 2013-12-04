package edu.stanford.nlp.bioprocess.joint.core;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.IntPair;

/**
 * 
 * @author svivek
 * 
 */
public class Input implements IInstance {
  public final Annotation annotation;
  public final int id;
  
  private final int[] triggerCandidates;
  private final IntPair[][] argumentCandidates;
  private final IntPair[] relationCandidates;

  public Input(Annotation annotation, int id) {
    this.annotation = annotation;
    this.id = id;

    triggerCandidates = createTriggerCandidates(annotation);

    argumentCandidates = createArgumentCandidates(annotation, triggerCandidates);
    assert triggerCandidates.length == argumentCandidates.length;

    relationCandidates = createRelationCandidates(annotation, triggerCandidates);
  }

  public int getNumberOfTriggers() {
    return triggerCandidates.length;
  }

  public int getTriggerTokenId(int triggerId) {
    return triggerCandidates[triggerId];
  }

  public int getNumberOfArgumentCandidates(int triggerId) {
    return argumentCandidates[triggerId].length;
  }

  public IntPair getArgumentCandidateSpan(int triggerId, int candidateId) {
    return argumentCandidates[triggerId][candidateId];
  }

  public int getNumberOfEERelationCandidates() {
    return relationCandidates.length;
  }

  public IntPair getEERelationCandidatePair(int eeRelationCandidateId) {
    return relationCandidates[eeRelationCandidateId];
  }

  @Override
  public String toString() {
    // TODO Auto-generated method stub
    throw new RuntimeException("Input.toString() not yet implemented!");
  }

  /**
   * Uses the annotation and the trigger candidates to create a set of plausible
   * event-event relation candidates.
   * 
   * @param annotation
   * @param triggerCandidates
   * @return An array of pairs of trigger candidates, representing event-event
   *         relations
   */
  private IntPair[] createRelationCandidates(Annotation annotation,
      int[] triggerCandidates) {
    // TODO Auto-generated method stub
    return null;
  }

  /**
   * Create a set of possible arguments for each trigger
   * 
   * @param annotation
   * @param triggerCandidates
   * @return An two dimensional array of pairs of integers that contains, for
   *         each event trigger (the first dimension of the output), an array of
   *         argument candidate spans.
   */
  private IntPair[][] createArgumentCandidates(Annotation annotation,
      int[] triggerCandidates) {
    // TODO Auto-generated method stub
    return null;
  }

  /**
   * Create a set of plausible trigger candidates.
   * 
   * @param annotation
   * @return An array of integers. Each integer is the token id of a possible
   *         event trigger.
   */
  private int[] createTriggerCandidates(Annotation annotation) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public double size() {
    // ignore this for now
    return 1;
  }

  /**
   * Check if (trigger1, trigger2) or (trigger2, trigger1) is a valid relation
   * candidate.
   * 
   * @param trigger1
   * @param trigger2
   * @return
   */
  public boolean isRelationCandidate(int trigger1, int trigger2) {
    // TODO
    return false;
  }
}
