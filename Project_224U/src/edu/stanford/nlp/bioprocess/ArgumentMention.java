package edu.stanford.nlp.bioprocess;

import java.util.List;

//import edu.stanford.nlp.bioprocess.Enumerators.MentionType;
import edu.stanford.nlp.ie.machinereading.structure.ExtractionObject;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;

/**
 * Each argument to an event
 * 
 * @author Aju
 */
public class ArgumentMention extends ExtractionObject {

  private static final long serialVersionUID = -2745903102654191527L;

  /** Mention type, if available, e.g., nominal */
  private String corefID = "-1";
  
  private Tree argumentRoot;
  /** 
   * Offsets the head span, e.g., "George Bush" in the extent "the president George Bush"
   * The offsets are relative to the sentence containing this mention 
   */
  private Span headTokenSpan;

  /**
   * Position of the syntactic head word of this mention, e.g., "Bush" for the head span "George Bush"
   * The offset is relative the sentence containing this mention
   * Note: use headTokenSpan when sequence tagging entity mentions not this. 
   *       This is meant to be used only for event/relation feature extraction! 
   */
  private int syntacticHeadTokenPosition;
  
  private IndexedWord headInDependencyTree;
  
  private String normalizedName;

  public ArgumentMention(String objectId,
      CoreMap sentence,
      Span extentSpan)
      //Span headSpan,
      //String type,
      //String subtype,
      //MentionType mentionType)
      {
    super(objectId, sentence, extentSpan, "", "");
   
    this.headTokenSpan = null;
    this.syntacticHeadTokenPosition = -1;
    this.normalizedName = null;
  }
  
  public void setHeadInDependencyTree(IndexedWord head) {
	  this.headInDependencyTree = head;
  }
  
  public IndexedWord getHeadInDependencyTree(){
	  return this.headInDependencyTree;
  }

  public void setTreeNode(Tree node) {
	  argumentRoot = node;
  }
  
  public Tree getTreeNode() {
	  return argumentRoot;
  }
  
  public String getCorefID(){
    return corefID;
  }

  public void setCorefID(String id) {
    this.corefID = id;
  }
  //public MentionType getMentionType() { return mentionType; }

  public Span getHead() { return headTokenSpan; }

  public int getHeadTokenStart() {
    return headTokenSpan.start();
  }

  public int getHeadTokenEnd() {
    return headTokenSpan.end();
  }

  public void setHeadTokenSpan(Span s) {
    headTokenSpan = s;
  }

  public CoreLabel getHeadToken() {
	  //LogInfo.logs(this.getHeadTokenStart());
	  return this.getSentence().get(TokensAnnotation.class).get(this.getHeadTokenStart());
  }
  
  public void setHeadTokenPosition(int i) {
    this.syntacticHeadTokenPosition = i;
  }

  public int getSyntacticHeadTokenPosition() {
    return this.syntacticHeadTokenPosition;
  }

  public CoreLabel getSyntacticHeadToken() {
    List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
    return tokens.get(syntacticHeadTokenPosition);
  }

  public Tree getSyntacticHeadTree() {
    Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
    return tree.getLeaves().get(syntacticHeadTokenPosition);
  }
  
  public String getNormalizedName() { return normalizedName; }
  public void setNormalizedName(String n) { normalizedName = n; }

  public boolean  equals(ArgumentMention other) {
	  if(this.getHead().equals(other.getHead()) && this.getSentence().equals(other.getSentence()))
		  return true;
	  return false;
  }
  
  private static int MENTION_COUNTER = 0;

  /**
   * Creates a new unique id for an entity mention
   * @return the new id
   */
  public static synchronized String makeUniqueId() {
    MENTION_COUNTER ++;
    return "EntityMention-" + MENTION_COUNTER;
  }
}

