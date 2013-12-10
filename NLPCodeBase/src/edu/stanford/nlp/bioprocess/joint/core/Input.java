package edu.stanford.nlp.bioprocess.joint.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;

/**
 * 
 * @author svivek
 * 
 */
public class Input implements IInstance, Serializable {

  private static final long serialVersionUID = 7286517679265381516L;
  public final Annotation annotation;
  public final String id;

  private final int[] triggerCandidates;
  private final IntPair[][] argumentCandidates;
  private final IntPair[] relationCandidates;

  // what follows are bad implementations because they use maps of primitives.
  // But if this isn't a bottleneck, we can keep this. Otherwise, we could move
  // to an efficient primitive map, like Trove

  /**
   * A map from trigger token to trigger id
   */
  private final Map<Integer, Integer> triggerCandidateToId;

  /**
   * A list of maps, one for each trigger, each of which is a map from the
   * argument span to the argument candidate id for that trigger
   * 
   */
  private final List<Map<IntPair, Integer>> argumentCandidateToId;

  /**
   * A map from relation candidate to its id
   */
  private final Map<IntPair, Integer> relationCandidateToId;

  /**
   * A local cache of the tokens. This gets accessed frequently in different
   * methods.
   */
  private final List<CoreLabel> tokens;

  public Input(Annotation annotation, String id) {
    this.annotation = annotation;
    this.id = id;

    triggerCandidates = createTriggerCandidates(annotation);

    argumentCandidates = createArgumentCandidates(annotation, triggerCandidates);

    relationCandidates = createRelationCandidates(annotation, triggerCandidates);

    validateData();

    this.triggerCandidateToId = Collections
        .unmodifiableMap(getTriggersInverseMap());

    this.argumentCandidateToId = Collections
        .unmodifiableList(getArgumentCandidateInverseMaps());

    this.relationCandidateToId = Collections
        .unmodifiableMap(getRelationCandidateInverseMap());
    tokens = annotation.get(TokensAnnotation.class);
  }

  private Map<IntPair, Integer> getRelationCandidateInverseMap() {
    Map<IntPair, Integer> map = new HashMap<>();
    for (int i = 0; i < relationCandidates.length; i++) {
      map.put(relationCandidates[i], i);
    }
    return map;
  }

  private Map<Integer, Integer> getTriggersInverseMap() {
    Map<Integer, Integer> triggerCandidateToId = new HashMap<>();
    for (int i = 0; i < triggerCandidates.length; i++) {
      triggerCandidateToId.put(triggerCandidates[i], i);
    }
    return triggerCandidateToId;
  }

  private List<Map<IntPair, Integer>> getArgumentCandidateInverseMaps() {
    List<Map<IntPair, Integer>> argumentCandidateToId = new ArrayList<>();
    for (int i = 0; i < triggerCandidates.length; i++) {
      HashMap<IntPair, Integer> map = new HashMap<>();

      for (int candidateId = 0; candidateId < argumentCandidates[i].length; candidateId++) {
        map.put(argumentCandidates[i][candidateId], candidateId);
      }
      argumentCandidateToId.add(Collections.unmodifiableMap(map));
    }
    return argumentCandidateToId;
  }

  private void validateData() {
    // the number of trigger candidates should be equal to the number of
    // argument candidates
    assert triggerCandidates.length == argumentCandidates.length : "Invalid number of argument candidates.";

    int numTokens = annotation.get(TokensAnnotation.class).size();

    // each argument candidate of each trigger should be a valid token.
    for (int triggerId = 0; triggerId < triggerCandidates.length; triggerId++) {
      for (IntPair span : argumentCandidates[triggerId]) {
        assert span.getSource() >= 0 && span.getSource() <= numTokens : "Invalid tokens in "
            + span;

        assert span.getTarget() >= 0 && span.getTarget() <= numTokens : "Invalid tokens in "
            + span;
      }
    }

    // each relation candidate should connect two triggers
    for (int i = 0; i < relationCandidates.length; i++) {
      assert isValidTriggerId(relationCandidates[i].getSource());
      assert isValidTriggerId(relationCandidates[i].getTarget());
    }

    // TODO: Check that the relation candidates have at least one connected
    // component
  }

  public int getNumberOfTriggers() {
    return triggerCandidates.length;
  }

  public int getTriggerTokenId(int triggerId) {
    assert isValidTriggerId(triggerId);
    return triggerCandidates[triggerId];
  }

  public int getTriggerIndex(int tokenIndex) {
    if (!this.triggerCandidateToId.containsKey(tokenIndex))
      throw new RuntimeException("Invalid token index " + tokenIndex);
    return this.triggerCandidateToId.get(tokenIndex);
  }

  public int getNumberOfArgumentCandidates(int triggerId) {
    return argumentCandidates[triggerId].length;
  }

  public IntPair getArgumentCandidateSpan(int triggerId, int candidateId) {
    assert isValidTriggerId(triggerId);

    if (candidateId < 0 || candidateId >= argumentCandidates[triggerId].length)
      throw new RuntimeException("Invalid candidate id: " + candidateId);

    return argumentCandidates[triggerId][candidateId];
  }

  public int getArgumentSpanIndex(int triggerId, IntPair span) {
    assert isValidTriggerId(triggerId);

    Map<IntPair, Integer> argsForTrigger = this.argumentCandidateToId
        .get(triggerId);

    if (!argsForTrigger.containsKey(span))
      throw new RuntimeException("Invalid argument candidate " + span
          + " for trigger id" + triggerId);
    return argsForTrigger.get(span);
  }

  public int getNumberOfEERelationCandidates() {
    return relationCandidates.length;
  }

  public IntPair getEERelationCandidatePair(int eeRelationCandidateId) {
    return relationCandidates[eeRelationCandidateId];
  }

  public int getEERelationIndex(int trigger1, int trigger2) {

    assert isValidTriggerId(trigger1);
    assert isValidTriggerId(trigger2);

    if (this.relationCandidateToId.containsKey(new IntPair(trigger1, trigger2)))
      return relationCandidateToId.get(new IntPair(trigger1, trigger2));
    else
      throw new RuntimeException("(" + trigger1 + ", " + trigger2
          + ") is not a valid relation candidate");
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
    // TODO reimplement
    IntPair[] res = new IntPair[(triggerCandidates.length * (triggerCandidates.length - 1))/2];
    int index = 0;
    for (int i = 0; i < triggerCandidates.length - 1; i++) {
      for (int j = i + 1; j < triggerCandidates.length; j++) {
        res[index++] = new IntPair(i, j);
      }
    }
      
    /*IntPair[] res = new IntPair[(triggerCandidates.length * (triggerCandidates.length - 1))];
    int index = 0;
    for (int i = 0; i < triggerCandidates.length - 1; ++i) {
      for (int j = i + 1; j < triggerCandidates.length; ++j) {
        res[index++] = new IntPair(i, j);
        res[index++] = new IntPair(j, i);
      }
    }*/
    return res;
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
    IntPair[][] res = new IntPair[triggerCandidates.length][];
    List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);

    for (int i = 0; i < res.length; i++) {
      int tokenId = getTriggerTokenId(i);
      List<IntPair> entities = new ArrayList<IntPair>();
      CoreMap sentence = AnnotationUtils.getContainingSentence(sentences,
          tokenId, tokenId);
      for (Tree node : sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
        if (node.isLeaf() || node.value().equals("ROOT"))
          continue;
        entities.add(node.getSpan());
      }
      res[i] = new IntPair[entities.size()];
      for(int j=0; j<entities.size(); j++){
        res[i][j] = entities.get(j);
      }
    }

    /*for (int i = 0; i < res.length; ++i) {
      IntPair[] spans = new IntPair[((triggerCandidates.length + 1) * (triggerCandidates.length)) / 2];
      int index = 0;
      for (int j = 0; j < triggerCandidates.length; ++j) {
        for (int k = j + 1; k < triggerCandidates.length + 1; ++k) {
          spans[index++] = new IntPair(j, k);
        }
      }
      res[i] = spans;
    }*/
    return res;

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

    List<Tree> eventNodes = new ArrayList<Tree>();
    for (CoreMap sentence : annotation.get(SentencesAnnotation.class)) {
      for (Tree node : sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
        String value = node.value();
        if (node.isLeaf()
            || value.equals("ROOT")
            || !node.isPreTerminal()
            || !(value.startsWith("JJR") || value.startsWith("JJS")
                || value.startsWith("NN") || value.equals("JJ") || value
                  .startsWith("VB")))
          continue;
        eventNodes.add(node);
        // count++;
      }
    }

    int[] res = new int[eventNodes.size()];
    for (int i = 0; i < eventNodes.size(); i++) {
      res[i] = eventNodes.get(i).getSpan().getSource();
    }

    /*
     * int[] res = new int[annotation.get(TokensAnnotation.class).size()]; for
     * (int i = 0; i < res.length; ++i) {
     * 
     * res[i] = i; }
     */
    return res;
  }

  @Override
  public String toString() {
    // A very verbose toString method. Maybe this could be made better.

    StringBuffer sb = new StringBuffer();

    for (int triggerId = 0; triggerId < this.getNumberOfTriggers(); triggerId++) {
      CoreLabel triggerCoreLabel = getTriggerCoreLabel(triggerId);
      sb.append("Event trigger: " + triggerCoreLabel.originalText()
          + " (token index" + triggerCoreLabel.index() + ")\n");

      sb.append("Argument candidates: \n");

      for (int argId = 0; argId < this.getNumberOfArgumentCandidates(triggerId); argId++) {

        sb.append("\t");
        for (CoreLabel c : getArgumentCandidate(triggerId, argId))
          sb.append(c.originalText() + " ");
        sb.append("\n");

      }
      sb.append("\n");
    }

    sb.append("Event-event relation candidates:\n");
    for (IntPair ee : this.relationCandidates) {
      int s = triggerCandidates[ee.getSource()];
      int t = triggerCandidates[ee.getTarget()];
      sb.append("\t" + tokens.get(s).originalText() + "\t"
          + tokens.get(t).originalText() + "\n");
    }

    return sb.toString();
  }

  /**
   * Gets the core label for this trigger
   * 
   * @param triggerId
   * @return
   */
  public CoreLabel getTriggerCoreLabel(int triggerId) {
    assert isValidTriggerId(triggerId);
    return tokens.get(triggerCandidates[triggerId]);
  }

  /**
   * Gets the sequence of core labels for the specified argument candidate
   * 
   * @param trigger
   * @param candidateId
   * @return
   */
  public List<CoreLabel> getArgumentCandidate(int trigger, int candidateId) {

    assert isValidTriggerId(trigger);
    assert candidateId >= 0 && candidateId < argumentCandidates[trigger].length;

    IntPair span = argumentCandidates[trigger][candidateId];

    return new ArrayList<>(tokens.subList(span.getSource(), span.getTarget()));

  }

  @Override
  public double size() {
    // ignore this for now
    return 1;
  }

  /**
   * Checks if the triggerId is a valid one
   * 
   * @param triggerId
   * @return
   */
  public boolean isValidTriggerId(int triggerId) {
    return triggerId >= 0 && triggerId < triggerCandidates.length;
  }

  /**
   * Check if (triggerId1, triggerId2) or (triggerId2, triggerId1) is a valid
   * relation candidate.
   * 
   * @param triggerId1
   * @param triggerId2
   * @return
   */
  public boolean isRelationCandidate(int triggerId1, int triggerId2) {
    assert isValidTriggerId(triggerId1);
    assert isValidTriggerId(triggerId2);

    return this.relationCandidateToId.containsKey(new IntPair(triggerId1,
        triggerId2));
  }

  /**
   * Checks if the span is an argument candidate for the specified triggerId
   * 
   * @param triggerId
   * @param span
   * @return
   */
  public boolean isArgumentCandidate(int triggerId, IntPair span) {
    assert isValidTriggerId(triggerId);

    Map<IntPair, Integer> argsForTrigger = this.argumentCandidateToId
        .get(triggerId);

    return argsForTrigger.containsKey(span);
  }
}
