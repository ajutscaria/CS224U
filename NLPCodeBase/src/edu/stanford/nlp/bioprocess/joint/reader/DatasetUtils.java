package edu.stanford.nlp.bioprocess.joint.reader;

import java.util.Arrays;
import java.util.List;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;

/**
 * static utils while reading the dataset
 * @author jonathanberant
 *
 */
public class DatasetUtils {

  public static final String TEXT_EXTENSION = ".txt";
  public static final String ANNOTATION_EXTENSION = ".ann";
  public static final String EVENT_TYPE = "Event";
  public static final String ENTITY_TYPE = "Entity";
  public static final String STATIC_ENTITY_TYPE = "Static-Event";
  public static final String EVENT_LABEL = "E";
  public static final String OTHER_LABEL = "O";
  public static final String NEXT_EVENT = "next-event", COTEMPORAL_EVENT = "cotemporal", SAME_EVENT = "same-event",
      SUPER_EVENT = "super-event", ENABLES = "enables", CAUSE ="cause",
      DESTINATION = "destination", LOCATION = "location", THEME = "theme", RESULT = "result", 
      AGENT = "agent", ORIGIN = "origin", TIME = "time", RAW_MATERIAL = "raw-material";
  public static final List<String> Punctuations = Arrays.asList(".", ",");
  
  public static String PERMISSIBILE_SPAN="ALL";
  
  public static int[] mapCharBeginOffsetToTokenIndex(List<CoreLabel> tokens) {
    int[] res = new int[tokens.get(tokens.size()-1).get(CharacterOffsetBeginAnnotation.class)];
    Arrays.fill(res, -1);
    for(int i = 0; i < tokens.size(); ++i) {
      res[tokens.get(i).get(CharacterOffsetBeginAnnotation.class)]=i;
    }    
    return res;
  }

  public static int[] mapCharEndOffsetToTokenIndex(List<CoreLabel> tokens) {
    int[] res = new int[tokens.get(tokens.size()-1).get(CharacterOffsetEndAnnotation.class)];
    Arrays.fill(res, -1);
    for(int i = 0; i < tokens.size(); ++i) {
      res[tokens.get(i).get(CharacterOffsetEndAnnotation.class)]=i;
    }    
    return res;
  }

  public static boolean isEvent(String str) {
    return str.equals(EVENT_TYPE) || str.equals(STATIC_ENTITY_TYPE);
  }

  public static boolean isEventEventRelation(String str) {
    return str.startsWith(NEXT_EVENT) ||
        str.startsWith(COTEMPORAL_EVENT) ||
        str.startsWith(SAME_EVENT) ||
        str.startsWith(SUPER_EVENT) ||
        str.startsWith(ENABLES) ||
        str.startsWith(CAUSE);
  }

  public static boolean isRole(String str) {
    return str.startsWith(AGENT) ||
        str.startsWith(THEME) ||
        str.startsWith(RESULT) ||
        str.startsWith(ORIGIN) ||
        str.startsWith(DESTINATION) ||
        str.startsWith(RAW_MATERIAL) ||
        str.startsWith(LOCATION) ||
        str.startsWith(TIME);
  }

  public static String getLabel(String edgeLabel) {
    //EVENT-EVENT RELATIONS
    if(edgeLabel.startsWith(NEXT_EVENT)) return NEXT_EVENT;
    if(edgeLabel.startsWith(COTEMPORAL_EVENT)) return COTEMPORAL_EVENT;
    if(edgeLabel.startsWith(SAME_EVENT)) return SAME_EVENT;
    if(edgeLabel.startsWith(SUPER_EVENT)) return SUPER_EVENT;
    if(edgeLabel.startsWith(ENABLES)) return ENABLES;
    if(edgeLabel.startsWith(CAUSE)) return CAUSE;
    //ROLES
    if(edgeLabel.startsWith(AGENT)) return AGENT;
    if(edgeLabel.startsWith(THEME)) return THEME;
    if(edgeLabel.startsWith(RESULT)) return RESULT;
    if(edgeLabel.startsWith(ORIGIN)) return ORIGIN;
    if(edgeLabel.startsWith(DESTINATION)) return DESTINATION;
    if(edgeLabel.startsWith(RAW_MATERIAL)) return RAW_MATERIAL;
    if(edgeLabel.startsWith(LOCATION)) return LOCATION;
    if(edgeLabel.startsWith(TIME)) return TIME;

    else throw new RuntimeException("Illegal label: " + edgeLabel);
  }

  public static IntPair getSpan(int beginToken, int endToken) {
    if(PERMISSIBILE_SPAN.equals("ALL"))
      return new IntPair(beginToken,endToken);
    else if(PERMISSIBILE_SPAN.equals("NODE")) {
      return new IntPair(0,0);
    }
    throw new RuntimeException("Illegal type of permissible span: " + PERMISSIBILE_SPAN);
  }
  
  //// FROM HERE COPIED AJU'S CODE ////

  public static Tree getEntityNodeBest(CoreMap sentence, IntPair entitySpan) {
    Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
    for (Tree node : syntacticParse.preOrderNodeList()) {
      if(node.isLeaf())
        continue;

      IntPair span = node.getSpan();
      if(span.getSource() == entitySpan.getSource() && span.getTarget() == entitySpan.getTarget()-1) {
        if(node.value().equals("NN") || node.value().equals("PRP") || node.value().equals("NP") || node.value().equals("NNS"))
          return node;
      }
      if(span.getSource() == entitySpan.getSource() - 1 && span.getTarget() == entitySpan.getTarget() - 1) {
        //To check for an extra determiner like "a" or "the" in front of the entity
        String POSTag = sentence.get(TokensAnnotation.class).get(span.getSource()).get(PartOfSpeechAnnotation.class);
        if(POSTag.equals("DT") || POSTag.equals("PRP$")) {
          if(node.value().equals("NN") || node.value().equals("PRP") || node.value().equals("NP") || node.value().equals("NNS"))
            return node;
        }
      }
      if(span.getSource() == entitySpan.getSource() && span.getTarget() == entitySpan.getTarget()) {
        //To check for an extra punctuation at the end of the entity.
        List<Tree> leaves = node.getLeaves();
        if(Punctuations.contains(leaves.get(leaves.size()-1).toString())) {
          if(node.value().equals("NN") || node.value().equals("PRP") || node.value().equals("NP") || node.value().equals("NNS"))
            return node;
        }
      }
    }
    return null;  
  }
  
  public static Tree getEntityNode(CoreMap sentence, IntPair entitySpan) {

    // Perfect Match
    Tree bestMatch = getEntityNodeBest(sentence, entitySpan);
    if (bestMatch != null) {
      return bestMatch;
    }
  
    IntPair entitySpanNoLastToken = new IntPair(entitySpan.getSource(),entitySpan.getTarget()-1);
    while(entitySpanNoLastToken.getTarget() - entitySpanNoLastToken.getSource()!= 0) {
      //Remove last token
      bestMatch = getEntityNodeBest(sentence, entitySpanNoLastToken);
      if (bestMatch != null) {
        return bestMatch;
      }
      entitySpanNoLastToken = new IntPair(entitySpanNoLastToken.getSource(),entitySpanNoLastToken.getTarget()-1);
    }
    
    IntPair entitySpanNoFirstToken = new IntPair(entitySpan.getSource()+1,entitySpan.getTarget());
    while(entitySpanNoFirstToken.getTarget() - entitySpanNoFirstToken.getSource() != 0) {
      //Remove first token
      bestMatch = getEntityNodeBest(sentence, entitySpanNoFirstToken);
      if (bestMatch != null) {
        return bestMatch;
      }
      entitySpanNoFirstToken = new IntPair(entitySpanNoFirstToken.getSource()+1,entitySpanNoFirstToken.getTarget());
    }
    throw new RuntimeException("Could not find a node for sentence " + sentence + " and span " + entitySpan); 
  }
}
