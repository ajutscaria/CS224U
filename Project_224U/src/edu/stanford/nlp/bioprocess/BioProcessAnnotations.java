package edu.stanford.nlp.bioprocess;

import java.util.HashMap;
import java.util.List;

import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.ErasureUtils;

/**
 * Annotations specific to the bio process data structures
 * @author Aju
 *
 */
public class BioProcessAnnotations {
  private BioProcessAnnotations() {} // only static members
  
  /**
   * The CoreMap key for getting the entity mentions corresponding to a document.
   * 
   * This key is typically set on sentence annotations.
   */
  public static class EntityMentionsAnnotation implements CoreAnnotation<List<EntityMention>> {
    public Class<List<EntityMention>> getType() {
      return ErasureUtils.uncheckedCast(List.class);
    }
  }

  /**
   * The CoreMap key for getting the relation mentions corresponding to a document.
   * 
   * This key is typically set on sentence annotations.
   */
  /*public static class RelationMentionsAnnotation implements CoreAnnotation<List<ArgumentRelation>> {
    public Class<List<ArgumentRelation>> getType() {
      return ErasureUtils.<Class<List<ArgumentRelation>>>uncheckedCast(List.class);
    }
  } */
  
  /**
   * The CoreMap key for getting the event mentions corresponding to a document.
   * 
   * This key is typically set on sentence annotations.
   */
  public static class EventMentionsAnnotation implements CoreAnnotation<List<EventMention>> {
    public Class<List<EventMention>> getType() {
      return ErasureUtils.<Class<List<EventMention>>>uncheckedCast(List.class);
    }
  }
  
  /**
   * The CoreMap key for getting the TreeNode to CoreLabel mapping.
   * 
   * This key is typically set on sentence annotations.
   */
  public static class TreeNodeAnnotation implements CoreAnnotation<HashMap<Tree, CoreLabel>> {
    public Class<HashMap<Tree, CoreLabel>> getType() {
      return ErasureUtils.<Class<HashMap<Tree, CoreLabel>>>uncheckedCast(List.class);
    }
  }
}
