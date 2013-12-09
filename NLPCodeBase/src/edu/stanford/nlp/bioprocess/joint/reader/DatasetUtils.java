package edu.stanford.nlp.bioprocess.joint.reader;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import edu.stanford.nlp.bioprocess.Utils;
import edu.stanford.nlp.bioprocess.joint.core.Input;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import fig.basic.LogInfo;
import fig.prob.SampleUtils;

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

  public static String PERMISSIBILE_SPAN="NODE";

  public static int[] mapCharBeginOffsetToTokenIndex(List<CoreLabel> tokens) {
    int[] res = new int[tokens.get(tokens.size()-1).get(CharacterOffsetBeginAnnotation.class)+1];
    Arrays.fill(res, -1);
    for(int i = 0; i < tokens.size(); ++i) {
      int beginIndex = tokens.get(i).get(CharacterOffsetBeginAnnotation.class);
      res[beginIndex]=i;
    }    
    return res;
  }

  public static int[] mapCharEndOffsetToTokenIndex(List<CoreLabel> tokens) {
    int[] res = new int[tokens.get(tokens.size()-1).get(CharacterOffsetEndAnnotation.class)+1];
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

  public static IntPair getEntitySpan(Input input, IntPair span) {
    if(PERMISSIBILE_SPAN.equals("ALL"))
      return span;
    else if(PERMISSIBILE_SPAN.equals("NODE")) {
      //find the corrected node span
      Tree resNode=null;
      int offset = 0;
      List<CoreMap> sentences = input.annotation.get(SentencesAnnotation.class);
      for(CoreMap sentence: sentences) {
        if(span.getSource()>=offset && span.getTarget() <= offset+sentence.get(TokensAnnotation.class).size()) {
          resNode = getEntityNode(sentence, new IntPair(span.getSource()-offset,span.getTarget()-offset));
          IntPair resSpan = new IntPair(offset+resNode.getSpan().getSource(),
              offset+resNode.getSpan().getTarget()+1); //node spans are inclusive
          return resSpan;
        }       
        offset+=sentence.get(TokensAnnotation.class).size();
      }
      throw new RuntimeException("Could not find entity span");
    }
    throw new RuntimeException("Illegal type of permissible span: " + PERMISSIBILE_SPAN);
  }

  public static int getEventNodeIndex(Input input, IntPair span) {

    if(span.getSource()+1==span.getTarget()) { //to save time - all span 1 are nodes
      return span.getSource();
    }

    int offset = 0;
    List<CoreMap> paragraph = input.annotation.get(SentencesAnnotation.class);

    for(CoreMap sentence: paragraph) {
      if(span.getSource()>=offset && span.getTarget() <= offset+sentence.get(TokensAnnotation.class).size()) {
        Tree eventNode = getEventNode(sentence,new IntPair(span.getSource()-offset,span.getTarget()-offset));
        IndexedWord head = Utils.findDependencyNode(sentence, eventNode);
        return offset+(head.index()-1); //index in indexed word starts at 1 not 0
      }       
      offset+=sentence.get(TokensAnnotation.class).size();
    }
    throw new RuntimeException("Could not find event span for paragraph: " + paragraph);
  }

  public static <V> List<V> shuffle(List<V> examples, Random rand) {

    List<V> res = new ArrayList<V>();
    int[] perm = SampleUtils.samplePermutation(rand, examples.size());

    for(int i = 0 ; i < examples.size(); ++i) {
      res.add(examples.get(perm[i]));
    }
    return res;
  }

  private static Tree getNodeInIndex(CoreMap sentence, int index) {
    Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
    for (Tree node : syntacticParse.preOrderNodeList()) {
      if(node.isLeaf())
        continue;

      IntPair span = node.getSpan();
      if(span.getSource()==index && span.getTarget()==index)
        return node;
    }
    throw new RuntimeException("There should be a node for every index");
  }

  //// FROM HERE COPIED AJU'S CODE ////

  private static Tree getEntityNodeBest(CoreMap sentence, IntPair entitySpan) {
    Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
    for (Tree node : syntacticParse.preOrderNodeList()) {
      if(node.isLeaf())
        continue;

      IntPair span = node.getSpan();
      if(span.getSource() == entitySpan.getSource() && span.getTarget() == entitySpan.getTarget()-1) { //node spans are inclusive
        if(node.value().equals("NN") || node.value().equals("PRP") || node.value().equals("NP") || node.value().equals("NNS") 
            || node.value().equals("X"))
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
    //if nothing works - return the last word in the span
    bestMatch = getNodeInIndex(sentence, entitySpan.getTarget()-1);
    LogInfo.warnings("Returning the last node in span, sentence=%s, span=%s, node=%s",sentence, entitySpan, bestMatch);
    return bestMatch; 
  }


  public static Tree getEventNode(CoreMap sentence, IntPair eventSpan) {
    Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
    for(Tree node:syntacticParse.postOrderNodeList()) {
      if(node.isLeaf())
        continue;

      IntPair span = node.getSpan();
      if(span.getSource() == eventSpan.getSource() && span.getTarget() == eventSpan.getTarget()-1) {
        if(node.headPreTerminal(new CollinsHeadFinder()).value().equals("IN"))
          return getSingleEventNode(sentence, eventSpan);
        return node.headPreTerminal(new CollinsHeadFinder());
      }

      if(span.getSource() == eventSpan.getSource() - 1 && span.getTarget() == eventSpan.getTarget() - 1) {
        //To check for an extra determiner like "a" or "the" in front of the entity
        String POSTag = sentence.get(TokensAnnotation.class).get(span.getSource()).get(PartOfSpeechAnnotation.class);
        if(POSTag.equals("DT") || POSTag.equals("PRP$")) {
          return  node.headPreTerminal(new CollinsHeadFinder());
        }
      }
    }
    Tree ret = getSingleEventNode(sentence, eventSpan);
    if(ret!=null)
      return ret.headPreTerminal(new CollinsHeadFinder());
    throw new RuntimeException("Did not find event node for sentence: " + sentence + ", span="+eventSpan);
  }

  public static Tree getSingleEventNode(CoreMap sentence, IntPair eventSpan) {
    Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
    for(int i = eventSpan.getSource(); i < eventSpan.getTarget(); i++) {
      for(Tree node:syntacticParse.postOrderNodeList()) {
        if(node.isLeaf())
          continue;

        IntPair span = node.getSpan();
        if(span.getSource() == i && span.getTarget() == i && 
            ( (node.value().startsWith("VB") && !node.firstChild().value().equals("is") && !node.firstChild().value().equals("in")) || node.value().startsWith("NN"))) {
          return node;
        }
      }
    }
    //If everything fails, returns first pre-terminal
    for(Tree node:syntacticParse.postOrderNodeList()) {
      if(node.isLeaf())
        continue;

      IntPair span = node.getSpan();
      if(span.getSource() == eventSpan.getSource() && span.getTarget() == eventSpan.getSource()) {
        return node;
      }
    }
    return null;
  }

  public static CoreMap getContainingSentence(List<CoreMap> sentences, int begin, int end) {
    for(CoreMap sentence:sentences) {
      if(sentence.get(CharacterOffsetBeginAnnotation.class) <= begin && sentence.get(CharacterOffsetEndAnnotation.class) >= end)
        return sentence;
    }
    return null;
  }
}
