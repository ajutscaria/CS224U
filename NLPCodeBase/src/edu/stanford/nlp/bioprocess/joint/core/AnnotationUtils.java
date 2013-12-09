package edu.stanford.nlp.bioprocess.joint.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import edu.stanford.nlp.bioprocess.Utils;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.BeginIndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.EndIndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;

public class AnnotationUtils {
  
  private static final List<String> Punctuations = Collections.unmodifiableList(Arrays.asList(".", ","));
  
  public static CoreMap getContainingSentence(List<CoreMap> sentences,
      int begin, int end) {
    // LogInfo.logs(begin + ":" + end);
    for (CoreMap sentence : sentences) {
      if (sentence.get(BeginIndexAnnotation.class) <= begin
          && sentence.get(EndIndexAnnotation.class) >= end)
        return sentence;
    }
    return null;
  }

  public static Tree getSingleEventNode(CoreMap sentence, int eventToken) {
    Tree syntacticParse = sentence
        .get(TreeCoreAnnotations.TreeAnnotation.class);
    for (int spanStart = eventToken; spanStart < eventToken + 1; spanStart++) {
      for (Tree node : syntacticParse.postOrderNodeList()) {
        if (node.isLeaf())
          continue;

        IntPair span = node.getSpan();
        if (span.getSource() == spanStart
            && span.getTarget() == spanStart
            && ((node.value().startsWith("VB")
                && !node.firstChild().value().equals("is") && !node
                .firstChild().value().equals("in")) || node.value().startsWith(
                    "NN"))) {
          // LogInfo.logs("Compressing " + event.getValue() + " to " + node);
          return node;
        }
      }
    }
    // If everything fails, returns first pre-terminal
    for (Tree node : syntacticParse.postOrderNodeList()) {
      if (node.isLeaf())
        continue;

      IntPair span = node.getSpan();
      if (span.getSource() == eventToken && span.getTarget() == eventToken) {
        // LogInfo.logs("Compressing " + event.getValue() + " to " + node);
        return node;
      }
    }

    return null;
  }

  public static Tree getEventNode(CoreMap sentence, int eventToken) {
    Tree syntacticParse = sentence
        .get(TreeCoreAnnotations.TreeAnnotation.class);
    Span entitySpan = new Span(eventToken, eventToken + 1); // single word of
    // event
    for (Tree node : syntacticParse.postOrderNodeList()) {
      if (node.isLeaf())
        continue;

      IntPair span = node.getSpan();
      if (span.getSource() == entitySpan.start()
          && span.getTarget() == entitySpan.end() - 1) {
        if (node.headPreTerminal(new CollinsHeadFinder()).value().equals("IN"))
          return getSingleEventNode(sentence, eventToken);
        return node.headPreTerminal(new CollinsHeadFinder());
      }

      if (span.getSource() == entitySpan.start() - 1
          && span.getTarget() == entitySpan.end() - 1) {
        // To check for an extra determiner like "a" or "the" in front of the
        // entity
        String POSTag = sentence.get(TokensAnnotation.class)
            .get(span.getSource()).get(PartOfSpeechAnnotation.class);
        if (POSTag.equals("DT") || POSTag.equals("PRP$")) {
          return node.headPreTerminal(new CollinsHeadFinder());
        }
      }
    }
    Tree ret = getSingleEventNode(sentence, eventToken);
    if (ret != null)
      return ret.headPreTerminal(new CollinsHeadFinder());

    // syntacticParse.pennPrint();
    LogInfo.logs("No EVENT match found!");
    return null;
  }

  public static Tree getEntityNodeBest(CoreMap sentence, IntPair entityspan) {
    Tree syntacticParse = sentence
        .get(TreeCoreAnnotations.TreeAnnotation.class);
    // syntacticParse.setSpans();
    Span entitySpan = new Span(entityspan.getSource(), entityspan.getTarget());
    for (Tree node : syntacticParse.preOrderNodeList()) {
      if (node.isLeaf())
        continue;

      IntPair span = node.getSpan();
      if (span.getSource() == entitySpan.start()
          && span.getTarget() == entitySpan.end() - 1) {
        // LogInfo.logs(node.value());
        // LogInfo.logs(entity.getValue() + "| Found match - " + node);
        if (node.value().equals("NN") || node.value().equals("PRP")
            || node.value().equals("NP") || node.value().equals("NNS"))
          return node;
      }
      if (span.getSource() == entitySpan.start() - 1
          && span.getTarget() == entitySpan.end() - 1) {
        // To check for an extra determiner like "a" or "the" in front of the
        // entity
        String POSTag = sentence.get(TokensAnnotation.class)
            .get(span.getSource()).get(PartOfSpeechAnnotation.class);
        if (POSTag.equals("DT") || POSTag.equals("PRP$")) {
          // LogInfo.logs(entity.getValue() + "| Found match - " + node);
          if (node.value().equals("NN") || node.value().equals("PRP")
              || node.value().equals("NP") || node.value().equals("NNS"))
            return node;
        }
      }
      if (span.getSource() == entitySpan.start()
          && span.getTarget() == entitySpan.end()) {
        // To check for an extra punctuation at the end of the entity.
        List<Tree> leaves = node.getLeaves();
        if (Punctuations.contains(leaves.get(leaves.size() - 1).toString())) {
          // LogInfo.logs(entity.getValue() + "| Found match - " + node);
          if (node.value().equals("NN") || node.value().equals("PRP")
              || node.value().equals("NP") || node.value().equals("NNS"))
            return node;
        }
      }
    }
    return null;
  }

  public static Tree getEntityNode(CoreMap sentence, IntPair entityspan) {
    // Tree syntacticParse =
    // sentence.get(TreeCoreAnnotations.TreeAnnotation.class);

    // Perfect Match
    Tree bestMatch = getEntityNodeBest(sentence, entityspan);
    if (bestMatch != null) {
      return bestMatch;
    }

    IntPair entityNoLastToken = new IntPair(entityspan.getSource(),
        entityspan.getTarget() - 1);
    while (entityNoLastToken.getTarget() - entityNoLastToken.getSource() != 0) {
      // Remove last token
      bestMatch = getEntityNodeBest(sentence, entityNoLastToken);
      if (bestMatch != null) {
        // LogInfo.logs(entity.getValue() + "| Found match - " + bestMatch);
        return bestMatch;
      }
      entityNoLastToken = new IntPair(entityspan.getSource(),
          entityspan.getTarget() - 1);
    }
    // LogInfo.logs("Missed second section");
    IntPair entityNoFirstToken = new IntPair(entityspan.getSource() + 1,
        entityspan.getTarget());
    while (entityNoFirstToken.getTarget() - entityNoFirstToken.getSource() != 0) {
      // Remove first token

      bestMatch = getEntityNodeBest(sentence, entityNoFirstToken);
      if (bestMatch != null) {
        // LogInfo.logs(entity.getValue() + "| Found match - " + bestMatch);
        return bestMatch;
      }
      entityNoFirstToken = new IntPair(entityspan.getSource() + 1,
          entityspan.getTarget());
    }

    LogInfo.logs("No ENTITY match found!");

    // syntacticParse.pennPrint();
    return null;
  }

  public static List<Pair<String, String>> findWordsInBetween(Input input,
      Tree event1, Tree event2) {
    // TODO Auto-generated method stub
    List<Pair<String, String>> words = new ArrayList<Pair<String, String>>();
    boolean beginGettingWords = false;
    for (CoreMap sentence : input.annotation.get(SentencesAnnotation.class)) {
      for (Tree node : sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
        if (node.isPreTerminal()) {
          if (node == event1) {
            beginGettingWords = true;
          } else if (node == event2) {
            beginGettingWords = false;
          } else if (beginGettingWords) {
            // System.out.println(getText(node));
            IndexedWord indexedWord = Utils.findDependencyNode(sentence, node);
            // System.out.println(indexedWord);
            // System.out.println(node);
            // if(indexedWord != null)
            // words.add(new Pair<String, String>(indexedWord.lemma(),
            // node.value()));
            words.add(new Pair<String, String>(Utils.getText(node), node
                .value()));
          }
        }
      }
    }
    return words;
  }


  public static Pair<Integer, Integer> findNumberOfSentencesAndWordsBetween(
      Input input, Tree event1, Tree event2) {
    // TODO Auto-generated method stub
    int sentenceCount = 0, wordCount = 0;
    boolean beginGettingWords = false;
    for (CoreMap sentence : input.annotation.get(SentencesAnnotation.class)) {
      for (Tree node : sentence.get(TreeCoreAnnotations.TreeAnnotation.class)) {
        if (node.isPreTerminal()) {
          if (node == event1) {
            beginGettingWords = true;
          } else if (node == event2) {
            beginGettingWords = false;
          } else if (beginGettingWords) {
            wordCount++;
          }
        }
      }
      if (beginGettingWords)
        sentenceCount++;
    }
    return new Pair<Integer, Integer>(sentenceCount, wordCount);
  }
  
  public static boolean isFirstEventInSentence(Input input, int trig1) {
    List<CoreMap> sentences = input.annotation.get(SentencesAnnotation.class);
    int tokenId1 = input.getTriggerTokenId(trig1);
    CoreMap sentence1 = getContainingSentence(sentences, tokenId1, tokenId1);
    CoreMap lastSentence = null;
    for (int i = 0; i < input.getNumberOfTriggers(); i++) {
      if (i == trig1) {
        if (lastSentence != null && lastSentence.equals(sentence1)) {
          return false;
        }
        return true;
      }
      int itoken = input.getTriggerTokenId(i);
      lastSentence = getContainingSentence(sentences, itoken, itoken);
    }
    return false;
  }

}
