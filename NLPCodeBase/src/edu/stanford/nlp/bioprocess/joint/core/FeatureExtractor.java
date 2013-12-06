package edu.stanford.nlp.bioprocess.joint.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.bioprocess.Utils;
import edu.stanford.nlp.bioprocess.ArgumentRelation.RelationType;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreAnnotations.BeginIndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.EndIndexAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import fig.basic.LogInfo;

/**
 * 
 * @author heatherchen1003 TODO: Make sure what is trigger token. -> span(token,
 *         token+1)? TODO: Discuss about the domains of features
 * 
 */
public class FeatureExtractor {
  private static boolean useLexicalFeatures = true; //should be an option
  private static boolean useBaselineFeaturesOnly = true, runGlobalModel = false;
  
  private static final List<String> Punctuations = Collections.unmodifiableList(Arrays.asList(".", ","));
  private static final Set<String> nominalizations = Utils.getNominalizedVerbs();
  private static final HashMap<String, String> verbForms = Utils.getVerbForms();
  private static final HashMap<String, Integer> clusters = Utils.loadClustering();
  
  private static final List<String> TemporalConnectives = Arrays.asList(new String[] {
      "before", "after", "since", "when", "meanwhile", "lately", "include",
      "includes", "including", "included", "first", "begin", "begins", "began",
      "beginning", "begun", "start", "starts", "started", "starting", "lead",
      "leads", "causes", "cause", "result", "results", "then", "subsequently",
      "previously", "next", "later", "subsequent", "previous" });

  static List<String> diffClauseRelations = Arrays
      .asList(new String[] { "acomp", "advcl", "ccomp", "csubj", "infmod",
          "prepc", "purpcl", "xcomp" });
  static HashMap<String, String> MarkAndPPClusters = new HashMap<String, String>();
  static HashMap<String, String> AdvModClusters = new HashMap<String, String>();

  public static FeatureVector getTriggerFV(Input input, int trigger) {
    List<CoreMap> sentences = input.annotation.get(SentencesAnnotation.class);
    int tokenId = input.getTriggerTokenId(trigger);
    CoreMap sentence = getContainingSentence(sentences, tokenId, tokenId);
    Tree event = getEventNode(sentence, tokenId);

    // LogInfo.logs("Current node's text - " + getText(event));
    FeatureVector fv = new FeatureVector(); // TODO
    List<String> features = new ArrayList<String>();
    String currentWord = event.value();
    Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
    SemanticGraph graph = sentence
        .get(CollapsedCCProcessedDependenciesAnnotation.class);
    CoreLabel token = Utils.findCoreLabelFromTree(sentence, event);
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    int currentTokenIndex = event.getSpan().getSource();

    IndexedWord word = Utils.findDependencyNode(sentence, event);
    Tree parent = event.parent(root);

    String parentCFGRule = parent.value() + "->";
    for (Tree n : parent.getChildrenAsList()) {
      parentCFGRule += n.value() + "|";
    }
    parentCFGRule = parentCFGRule.trim();

    if (useLexicalFeatures) {// lexical
      String text = token.lemma().toLowerCase();
      if (verbForms.containsKey(text)) {
        features.add("lemma=" + verbForms.get(text));
      } else {
        features.add("lemma=" + token.lemma().toLowerCase());
      }
      features.add("word=" + token.originalText());// lexical
      features.add("POSlemma=" + currentWord + "," + token.lemma());

      if (clusters.containsKey(text)) {
        features.add("clusterID=" + clusters.get(text));
        // LogInfo.logs(text + ", clusterID=" + clusters.get(text));
      }
      for (SemanticGraphEdge e : graph.getOutEdgesSorted(word)) {
        if (e.getRelation().toString().equals("advmod")
            && (currentWord.startsWith("VB") || nominalizations.contains(text)))
          features.add("advmod:" + e.getTarget());
        // LogInfo.logs("TIMEE : " + e.getRelation() + ":" + e.getTarget());
        // features.add("depedgein="+ e.getRelation() + "," +
        // e.getTarget().toString().split("-")[1]);//need to deal with mult
        // children same tag?
        // features.add("depedgeinword="+currentWord +"," + e.getRelation() +
        // "," + e.getSource().toString().split("-")[0] + ","+
        // e.getSource().toString().split("-")[1]);
        // LogInfo.logs("depedgeinword="+currentWord +"," + e.getRelation() +
        // "," + e.getSource().toString().split("-")[0] + ","+
        // e.getSource().toString().split("-")[1]);
      }

      if (nominalizations.contains(token.value())) {
        // LogInfo.logs("Adding nominalization - " + leaves.get(0));
        features.add("nominalization");
      }
    }

    features.add("ParentPOS=" + parent.value());// both
    features.add("path="
        + StringUtils.join(Trees.pathNodeToNode(root, event, root), ",")
            .replace("up-ROOT,down-ROOT,", ""));// syntactic
    features.add("POSparentrule=" + currentWord + "," + parentCFGRule);// both

    String consecutiveTypes = "";
    if (currentTokenIndex > 0)
      consecutiveTypes += tokens.get(currentTokenIndex - 1).get(
          PartOfSpeechAnnotation.class);
    consecutiveTypes += currentWord;
    if (currentTokenIndex < tokens.size() - 1)
      consecutiveTypes += tokens.get(currentTokenIndex + 1).get(
          PartOfSpeechAnnotation.class);
    features.add("consecutivetypes=" + consecutiveTypes);// ?

    features.add("bias");// ?

    return fv;

  }

  public static FeatureVector getArgumentFV(Input input, int trigger,
      int argument) {
    List<CoreMap> sentences = input.annotation.get(SentencesAnnotation.class);
    IntPair entityspan = input.getArgumentCandidateSpan(trigger, argument);
    CoreMap sentence = getContainingSentence(sentences, entityspan.getSource(),
        entityspan.getTarget());
    int eventtokenId = input.getTriggerTokenId(trigger);
    Tree event = getEventNode(sentence, eventtokenId);
    Tree entity = getEntityNode(sentence, entityspan);

    // Tree event = eventMention.getTreeNode();
    FeatureVector fv = new FeatureVector(); // TODO
    List<String> features = new ArrayList<String>();
    // List<Tree> leaves = entity.getLeaves();
    Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
    boolean dependencyExists = Utils.isNodesRelated(sentence, entity, event);
    String depPath = Utils.getDependencyPath(sentence, entity, event);
    Tree parent = entity.parent(root);
    String parentCFGRule = parent.value() + "->";
    for (Tree n : parent.getChildrenAsList()) {
      parentCFGRule += n.value() + "|";
    }
    parentCFGRule = parentCFGRule.trim();
    int containS = 0;
    for (Tree node : entity.getChildrenAsList()) {
      if (node.value().equals("S") || node.value().equals("SBAR")) {
        containS = 1;
        break;
      }
    }

    // features.add("EntContainsS="+containS);
    features.add("EvtLemma=" + event.getLeaves().get(0).value());// lexical
    features.add("EntCatDepRel=" + entity.value() + "," + dependencyExists);// syntactic
    features.add("EntHeadEvtPOS="
        + Utils.findCoreLabelFromTree(sentence, entity).lemma() + ","
        + event.preTerminalYield().get(0).value());// both
    features.add("EvtToEntDepPath="
        + ((depPath.equals("") || depPath.equals("[]")) ? 0 : depPath
            .split(",").length));// syntactic
    features.add("EntHeadEvtHead="
        + entity.headTerminal(new CollinsHeadFinder()) + ","
        + event.getLeaves().get(0)); // syntactic
    features.add("EntNPAndRelatedToEvt="
        + (entity.value().equals("NP") && Utils.isNodesRelated(sentence,
            entity, event)));// both
    features.add("bias");

    return fv;
  }

  public static FeatureVector getRelationFV(Input input, int trig1, int trig2) {
    List<CoreMap> sentences = input.annotation.get(SentencesAnnotation.class);
    int tokenId1 = input.getTriggerTokenId(trig1);
    CoreMap sentence1 = getContainingSentence(sentences, tokenId1, tokenId1);
    Tree event1 = getEventNode(sentence1, tokenId1);
    int tokenId2 = input.getTriggerTokenId(trig2);
    CoreMap sentence2 = getContainingSentence(sentences, tokenId2, tokenId2);
    Tree event2 = getEventNode(sentence2, tokenId2);
    ClusterSetup();

    FeatureVector fv = new FeatureVector();// TODO
    List<String> features = new ArrayList<String>();
    CoreLabel event1CoreLabel = Utils.findCoreLabelFromTree(sentence1, event1), event2CoreLabel = Utils
        .findCoreLabelFromTree(sentence2, event2);
    boolean isImmediatelyAfter = (trig2 == trig1 + 1) ? true : false;

    List<Pair<String, String>> wordsInBetween = findWordsInBetween(input,
        event1, event2);
    // Number of sentences and words between two event mentions. Quantized to
    // 'Low', 'Medium', 'High' etc.
    Pair<Integer, Integer> countsSentenceAndWord = findNumberOfSentencesAndWordsBetween(
        input, event1, event2);
    int sentenceBetweenEvents = countsSentenceAndWord.first();
    int wordsBetweenEvents = countsSentenceAndWord.second();

    SemanticGraph graph1 = sentence1
        .get(CollapsedCCProcessedDependenciesAnnotation.class);
    IndexedWord indexedWord1 = Utils.findDependencyNode(sentence1, event1);
    SemanticGraph graph2 = sentence2
        .get(CollapsedCCProcessedDependenciesAnnotation.class);
    IndexedWord indexedWord2 = Utils.findDependencyNode(sentence2, event2);

    String pos1 = event1.value(), pos2 = event2.value();

    // Lemmas of both events
    String lemma1 = event1CoreLabel.lemma().toLowerCase();
    if (verbForms.containsKey(lemma1)) {
      lemma1 = verbForms.get(lemma1);
    }
    String lemma2 = event2CoreLabel.lemma().toLowerCase();
    if (verbForms.containsKey(lemma2)) {
      lemma2 = verbForms.get(lemma2);
    }
    features.add("lemmas:" + lemma1 + "+" + lemma2);// lexical

    // Is event2 immediately after event1?
    if (!runGlobalModel) {
      features.add("isImmediatelyAfter:" + isImmediatelyAfter);// other
    }

    if (isImmediatelyAfter) {
      // Add words in between two event mentions if they are adjacent in text.
      StringBuffer phrase = new StringBuffer();
      for (int wordCounter = 0; wordCounter < wordsInBetween.size(); wordCounter++) {
        phrase.append(wordsInBetween.get(wordCounter).first + " ");
        String POS = wordsInBetween.get(wordCounter).second, word = wordsInBetween
            .get(wordCounter).first;
        String POS2 = wordCounter < wordsInBetween.size() - 1 ? wordsInBetween
            .get(wordCounter + 1).second : "", word2 = wordCounter < wordsInBetween
            .size() - 1 ? wordsInBetween.get(wordCounter + 1).first : "";

        if (!TemporalConnectives.contains(word.toLowerCase())) {
          if (POS.startsWith("VB") && POS2.equals("IN")) {
            features.add("wordsInBetween:" + word + " " + word2);// lexical
            wordCounter++;
          } else
            features.add("wordsInBetween:" + word);// lexical
        } else {
          if (sentenceBetweenEvents < 2) {
            // LogInfo.logs("TEMPORAL CONNECTIVE ADDED: " + example.id + " " +
            // lemma1 + " " + lemma2 + " " + word.toLowerCase());
            if (useBaselineFeaturesOnly) {
              features.add("temporalConnective:" + word.toLowerCase());// connective
            } else {
              features.add("connector:" + word.toLowerCase());// connective
              if (AdvModClusters.containsKey(word.toLowerCase())) {
                features.add("connectorCluster:"
                    + AdvModClusters.get(word.toLowerCase()));// connective
              }
            }
          }
        }
      }

    }

    if (!useBaselineFeaturesOnly) {
      if (isImmediatelyAfter) {
        // Is there an and within 5 words of each other
        if (wordsInBetween.size() <= 5) {
          for (int wordCounter = 0; wordCounter < wordsInBetween.size(); wordCounter++) {
            if (wordsInBetween.get(wordCounter).first.equals("and"))
              features.add("closeAndInBetween");// lexical
          }
        }

        // If event1 is the first event in the paragraph and is a
        // nominalization, it is likely that others are sub-events
        if (trig1 == 0 && event1.value().startsWith("NN")) {
          features.add("firstAndNominalization");// lexical
        }
      }
      // Are the lemmas same?
      features.add("eventLemmasSame:" + lemma1.equals(lemma2));// lexical

      // If second trigger is noun, the determiner related to it in dependency
      // tree.
      if (pos2.startsWith("NN")) {
        String determiner = Utils.getDeterminer(sentence2, event2);
        if (determiner != null) {
          features.add("determinerBefore2:" + determiner);// syntactic
        }
      }
    }

    // POS tags of both events
    features.add("POS:" + pos1 + "+" + pos2);// lexical
    features.add("numSentencesInBetween:"
        + quantizedSentenceCount(sentenceBetweenEvents));// lexical
    features.add("numWordsInBetween:" + quantizedWordCount(wordsBetweenEvents));// lexical

    // Features if the two triggers are in the same sentence.
    if (sentenceBetweenEvents == 0) {
      // Lowest common ancestor between the two event triggers. Reduces score.
      Tree root = sentence1.get(TreeCoreAnnotations.TreeAnnotation.class);
      Tree lca = Trees.getLowestCommonAncestor(event1, event2, root);
      features.add("lowestCommonAncestor:" + lca.value());// syntactic

      // Dependency path if the event triggers are in the same sentence.
      // LogInfo.logs(example.id + " " + lemma1 + " " + lemma2);
      String deppath = Utils.getUndirectedDependencyPath_Events(sentence1,
          event1, event2);
      if (!deppath.isEmpty()) {
        if (!useBaselineFeaturesOnly) {
          features.add("deppath:" + deppath);// syntactic
          features.add("deppathwithword:"
              + Utils.getUndirectedDependencyPath_Events_WithWords(sentence1,
                  event1, event2));// syntactic
        }
        // Does event1 dominate event2
        if (deppath.contains("->") && !deppath.contains("<-")) {
          features.add("1dominates2");// syntactic
        }

        if (deppath.contains("<-") && !deppath.contains("->")) {
          features.add("2dominates1");// syntactic
        }
      }

      // Extract mark relation
      // LogInfo.logs("Trying Marker: " + example.id + " " + lemma1 + " " +
      // lemma2 + " ");
      List<Pair<String, String>> markRelations = extractMarkRelation(input,
          sentence1, event1, event2, trig1);
      for (Pair<String, String> markRelation : markRelations) {
        // LogInfo.logs("MARKER ADDED: " + example.id + " " + lemma1 + " " +
        // lemma2 + " " + markRelation);
        if (useBaselineFeaturesOnly) {
          features.add("markRelation:" + markRelation.first());// mark
        } else {
          features.add("connector:" + markRelation.first());// mark
          // In some cases, we don't have clusters for some relation.
          if (!markRelation.second().isEmpty())
            features.add("connectorCluster:" + markRelation.second());// mark
        }
      }

      // Extract PP relation
      // LogInfo.logs("Trying PP: " + example.id + " " + lemma1 + " " + lemma2 +
      // " ");
      List<Pair<String, String>> ppRelations = extractPPRelation(input, trig1,
          sentence1, event1, event2);
      for (Pair<String, String> ppRelation : ppRelations) {
        // LogInfo.logs("PP ADDED: " + example.id + " " + lemma1 + " " + lemma2
        // + " " + ppRelation);
        if (useBaselineFeaturesOnly) {
          features.add("PPRelation:" + ppRelation.first());// pp
        } else {
          features.add("connector:" + ppRelation.first());// pp
          // In some cases, we don't have clusters (if we haven't included in
          // the list.
          if (!ppRelation.second().isEmpty()) {
            features.add("connectorCluster:" + ppRelation.second());// pp
          }
        }
      }
    }

    if (isImmediatelyAfter) {
      // Extract advmod relation
      // LogInfo.logs("Trying AdvMod: " + example.id + " " + lemma1 + " " +
      // lemma2 + " ");
      List<Pair<String, String>> advModRelations = extractAdvModRelation(
          sentence1, sentence2, event1, event2);
      for (Pair<String, String> advModRelation : advModRelations) {
        // LogInfo.logs("ADVMOD ADDED: " + example.id + " " + lemma1 + " " +
        // lemma2 + " " + advModRelation);
        if (useBaselineFeaturesOnly) {
          features.add("advModRelation:" + advModRelation.first());// advmod
        } else {
          features.add("connector:" + advModRelation.first());// advmod
          // In some cases, we don't have clusters for some relation.
          if (!advModRelation.second().isEmpty()) {
            features.add("connectorCluster:" + advModRelation.second());// advmod
          }
        }
      }
    }

    String advMod = extractAdvModRelation(graph2, indexedWord2);
    if (advMod != null && !advMod.isEmpty()) {
      features.add("advMod:" + advMod);// advmod
    }

    // See if the two triggers share a common lemma as child in the dependency
    // graph.
    List<Pair<IndexedWord, String>> w1Children = new ArrayList<Pair<IndexedWord, String>>();
    for (SemanticGraphEdge e : graph1.getOutEdgesSorted(indexedWord1)) {
      w1Children.add(new Pair<IndexedWord, String>(e.getTarget(), e
          .getRelation().toString()));
    }

    for (SemanticGraphEdge e : graph2.getOutEdgesSorted(indexedWord2)) {
      for (Pair<IndexedWord, String> pair : w1Children) {
        if (e.getTarget().originalText().equals(pair.first.originalText())) {
          // System.out.println(indexedWord1 + ":" + indexedWord2 +
          // " share children. " + pair.first.originalText()
          // +":" +pair.second+ "+" +e.getRelation().toString());
          // features.add("shareSameLemmaAsChild:"+ pair.second+ "+"
          // +e.getRelation().toString());
        }
        // System.out.println(indexedWord1 + ":" + indexedWord2 +
        // " share children. " + w.originalText());
        // features.add("shareSameLemmaAsChild")// + w.originalText());
      }
    }

    // Do they share a common argument in the dependency tree? (if they are in
    // the same sentence)
    if (sentenceBetweenEvents == 0) {
      List<SemanticGraphEdge> edges1 = graph1.getOutEdgesSorted(indexedWord1);
      List<SemanticGraphEdge> edges2 = graph2.getOutEdgesSorted(indexedWord2);
      for (SemanticGraphEdge e1 : edges1) {
        for (SemanticGraphEdge e2 : edges2) {
          if (e1.getTarget().equals(e2.getTarget())) {
            features.add("shareChild:" + e1.getRelation() + "+"
                + e2.getRelation());// syntactic
            break;
          }
        }
      }
    }
    features.add("bias");
    return fv;
  }

  public static FeatureVector getTriggerLabelFV(Input input, int trigger,
      boolean label) {
    FeatureVector fv = getTriggerFV(input, trigger);
    ArrayList<String> indicateFeatures = fv.getIndicateFeatures();
    ArrayList<fig.basic.Pair<String, Double>> generalFeatures = fv
        .getGeneralFeatures();
    for (String f : indicateFeatures) {
      f = label + "&" + f;
    }
    for (fig.basic.Pair<String, Double> f : generalFeatures) {
      f.setFirst(label + "&" + f.getFirst());
    }
    return fv;
  }

  public static FeatureVector getArgumentLabelFV(Input input, int trigger,
      int argument, String label) {
    FeatureVector fv = getArgumentFV(input, trigger, argument);
    ArrayList<String> indicateFeatures = fv.getIndicateFeatures();
    ArrayList<fig.basic.Pair<String, Double>> generalFeatures = fv
        .getGeneralFeatures();
    for (String f : indicateFeatures) {
      f = label + "&" + f;
    }
    for (fig.basic.Pair<String, Double> f : generalFeatures) {
      f.setFirst(label + "&" + f.getFirst());
    }
    return fv;
  }

  public static FeatureVector getRelationLabelFV(Input input, int trig1,
      int trig2, String label) {
    FeatureVector fv = getRelationFV(input, trig1, trig2);
    ArrayList<String> indicateFeatures = fv.getIndicateFeatures();
    ArrayList<fig.basic.Pair<String, Double>> generalFeatures = fv
        .getGeneralFeatures();
    for (String f : indicateFeatures) {
      f = label + "&" + f;
    }
    for (fig.basic.Pair<String, Double> f : generalFeatures) {
      f.setFirst(label + "&" + f.getFirst());
    }
    return fv;
  }

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

  private static void ClusterSetup() {
    MarkAndPPClusters.put("if", RelationType.PreviousEvent.toString());
    MarkAndPPClusters.put("until", RelationType.NextEvent.toString());
    MarkAndPPClusters.put("after", RelationType.PreviousEvent.toString());
    MarkAndPPClusters.put("as", RelationType.CotemporalEvent.toString());
    MarkAndPPClusters.put("because", RelationType.Causes.toString());
    MarkAndPPClusters.put("before", RelationType.NextEvent.toString());
    MarkAndPPClusters.put("since", RelationType.Causes.toString());
    MarkAndPPClusters.put("so", RelationType.Caused.toString());
    MarkAndPPClusters.put("while", RelationType.CotemporalEvent.toString());
    MarkAndPPClusters.put("during", RelationType.SuperEvent.toString());
    MarkAndPPClusters.put("upon", RelationType.PreviousEvent.toString());

    AdvModClusters.put("then", RelationType.PreviousEvent.toString());
    AdvModClusters.put("thus", RelationType.Causes.toString());
    AdvModClusters.put("also", RelationType.CotemporalEvent.toString());
    AdvModClusters.put("eventually", RelationType.PreviousEvent.toString());
    AdvModClusters.put("meanwhile", RelationType.CotemporalEvent.toString());
    AdvModClusters.put("thereby", RelationType.Causes.toString());
    AdvModClusters.put("finally", RelationType.PreviousEvent.toString());
    AdvModClusters.put("first", RelationType.SuperEvent.toString());
    AdvModClusters.put("hence", RelationType.Causes.toString());
    AdvModClusters.put("later", RelationType.PreviousEvent.toString());
    AdvModClusters.put("next", RelationType.PreviousEvent.toString());
    AdvModClusters.put("simultaneously",
        RelationType.CotemporalEvent.toString());
    AdvModClusters.put("subsequently", RelationType.PreviousEvent.toString());
    AdvModClusters.put("if", RelationType.NextEvent.toString());
    AdvModClusters.put("until", RelationType.PreviousEvent.toString());
    AdvModClusters.put("after", RelationType.NextEvent.toString());
    AdvModClusters.put("as", RelationType.CotemporalEvent.toString());
    AdvModClusters.put("because", RelationType.Caused.toString());
    AdvModClusters.put("so", RelationType.Causes.toString());
    AdvModClusters.put("result", RelationType.Causes.toString());
    AdvModClusters.put("results", RelationType.Causes.toString());
    AdvModClusters.put("lead", RelationType.Causes.toString());
    AdvModClusters.put("leads", RelationType.Causes.toString());
    AdvModClusters.put("cause", RelationType.Causes.toString());
    AdvModClusters.put("causes", RelationType.Causes.toString());
    AdvModClusters.put("while", RelationType.CotemporalEvent.toString());
    AdvModClusters.put("during", RelationType.SubEvent.toString());
    AdvModClusters.put("upon", RelationType.NextEvent.toString());
    AdvModClusters.put("include", RelationType.SuperEvent.toString());
    AdvModClusters.put("includes", RelationType.SuperEvent.toString());
    AdvModClusters.put("included", RelationType.SuperEvent.toString());
    AdvModClusters.put("including", RelationType.SuperEvent.toString());
    AdvModClusters.put("begin", RelationType.SuperEvent.toString());
    AdvModClusters.put("begins", RelationType.SuperEvent.toString());
    AdvModClusters.put("began", RelationType.SuperEvent.toString());
    AdvModClusters.put("begun", RelationType.SuperEvent.toString());
    AdvModClusters.put("beginning", RelationType.SuperEvent.toString());
    AdvModClusters.put("start", RelationType.SuperEvent.toString());
    AdvModClusters.put("starts", RelationType.SuperEvent.toString());
    AdvModClusters.put("started", RelationType.SuperEvent.toString());
    AdvModClusters.put("starting", RelationType.SuperEvent.toString());
    AdvModClusters.put("subsequent", RelationType.PreviousEvent.toString());
    AdvModClusters.put("previously", RelationType.NextEvent.toString());
    AdvModClusters.put("previous", RelationType.NextEvent.toString());
  }

  public static String quantizedSentenceCount(int numSentences) {
    if (numSentences == 0) {
      return "None";
    } else if (numSentences == 1) {
      return "Low";
    } else if (numSentences == 2 || numSentences == 3) {
      return "Medium";
    }
    return "High";
  }

  public static String quantizedWordCount(int numWords) {
    if (numWords <= 4) {
      return "Low";
    } else if (numWords <= 8) {
      return "Medium";
    } else if (numWords <= 15) {
      return "High";
    }
    return "VeryHigh";
  }

  private static List<Pair<String, String>> extractAdvModRelation(
      CoreMap sentence1, CoreMap sentence2, Tree event1, Tree event2) {

    SemanticGraph graph1 = sentence1
        .get(CollapsedCCProcessedDependenciesAnnotation.class);
    SemanticGraph graph2 = sentence2
        .get(CollapsedCCProcessedDependenciesAnnotation.class);
    IndexedWord indexedWord1 = Utils.findDependencyNode(sentence1, event1), indexedWord2 = Utils
        .findDependencyNode(sentence2, event2);
    List<Pair<String, String>> advModRelations = new ArrayList<Pair<String, String>>();
    if (indexedWord1 == null || indexedWord2 == null) {
      return advModRelations;
    }
    int event1Index = indexedWord1.index(), event2Index = indexedWord2.index();

    Pair<String, String> advModRelation1 = extractAdvModRelation(graph1,
        indexedWord1, indexedWord2, event1Index, event2Index);
    Pair<String, String> advModRelation2 = extractAdvModRelation(graph2,
        indexedWord2, indexedWord1, event1Index, event2Index);
    if (advModRelation1 != null) {
      // advModRelations.add(new Pair<String, String>(advModRelation1.first() +
      // "_1", advModRelation1.second()));
    }

    if (advModRelation2 != null) {
      advModRelations.add(new Pair<String, String>(advModRelation2.first(),
          advModRelation2.second()));
    }

    return advModRelations;
  }

  private static Pair<String, String> extractAdvModRelation(
      SemanticGraph graph, IndexedWord indexedWordThis,
      IndexedWord indexedWordThat, int event1Index, int event2Index) {
    for (SemanticGraphEdge e : graph.getOutEdgesSorted(indexedWordThis)) {
      if (e.getRelation().getShortName().equals("advmod")) {
        String advModName = e.getTarget().lemma().toLowerCase();

        // LogInfo.logs("AdvMod found '" + indexedWordThis + "':" + advModName);
        if (AdvModClusters.containsKey(advModName))
          return new Pair<String, String>(advModName,
              AdvModClusters.get(advModName));
        else
          return new Pair<String, String>(advModName, "");
      }
    }
    return null;
  }

  private static String extractAdvModRelation(SemanticGraph graph,
      IndexedWord indexedWord) {
    for (SemanticGraphEdge e : graph.getOutEdgesSorted(indexedWord)) {
      if (e.getRelation().getShortName().equals("advmod")) {
        return e.getTarget().originalText();
      }
    }
    return null;
  }

  private static List<Pair<String, String>> extractPPRelation(Input input,
      int trig1, CoreMap sentence1, Tree event1, Tree event2) {
    CoreMap sentence = sentence1;
    Tree root = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);

    int event1Index = event1.nodeNumber(root), event2Index = event2
        .nodeNumber(root);
    // Are the event triggers part of a Prepositional phrase individually (one
    // feature each)?

    List<Pair<String, String>> ppRelations = new ArrayList<Pair<String, String>>();

    Pair<String, String> ppRelation1 = extractPPRelation(input, trig1,
        sentence1, event1, event2, event1Index, event2Index);
    Pair<String, String> ppRelation2 = extractPPRelation(input, trig1,
        sentence1, event2, event1, event1Index, event2Index);
    if (ppRelation1 != null) {
      ppRelations.add(new Pair<String, String>(ppRelation1.first() + "_1",
          ppRelation1.second()));
    }

    if (ppRelation2 != null) {
      ppRelations.add(new Pair<String, String>(ppRelation2.first() + "_2",
          ppRelation2.second()));
    }

    return ppRelations;
  }

  private static Pair<String, String> extractPPRelation(Input input, int trig1,
      CoreMap sentence1, Tree thisEvent, Tree thatEvent, int event1Index,
      int event2Index) {
    Tree root = sentence1.get(TreeCoreAnnotations.TreeAnnotation.class);
    Tree node = thisEvent;
    // root.pennPrint();
    // Are the event triggers part of a Prepositional phrase individually (one
    // feature each)?
    while (!node.value().equals("ROOT") && !node.value().equals("S")
        && !node.value().equals("SBAR")) {
      if (node.value().equals("PP")) {
        for (Tree ponode : node.postOrderNodeList()) {
          if (ponode.isPreTerminal()
              && (ponode.value().equals("IN") || ponode.value().equals("TO"))) {
            // Other event should not be in the same PP
            if (node.dominates(thatEvent)) {
              // LogInfo.logs("PP dominates other event");
              return null;
            }
            Tree lca = Trees
                .getLowestCommonAncestor(thisEvent, thatEvent, root);
            // LogInfo.logs("ANCESTOR " + lca.value());
            List<String> path = Trees.pathNodeToNode(lca, ponode, lca);
            path.remove(path.size() - 1);
            path.remove(0);
            path.remove(0);

            // LogInfo.logs("PPPATH " + thisEvent + " " + path);
            // if(path.contains("down-S") || path.contains("down-SBAR")) {
            if (path.contains("down-S") || path.contains("down-SBAR")) {
              // LogInfo.logs("Contains S or SBAR");
              // return null;
            }

            path = Trees.pathNodeToNode(lca, thatEvent, lca);
            path.remove(path.size() - 1);
            path.remove(0);
            path.remove(0);

            // LogInfo.logs("PPPATH " + thatEvent + " " + path);
            if (path.contains("down-S") || path.contains("down-SBAR")) {
              // LogInfo.logs("Contains S or SBAR");
              // return null;
            }

            String ppName = ponode.firstChild().value().toLowerCase();
            int ppIndex = ponode.firstChild().nodeNumber(root);
            if (isFirstEventInSentence(input, trig1) && ppIndex < event1Index) {
              // LogInfo.logs("PP before :" +ppName);
              if (MarkAndPPClusters.containsKey(ppName))
                return new Pair<String, String>(ppName,
                    MarkAndPPClusters.get(ppName));
              else
                return new Pair<String, String>(ppName, "");
            } else if (ppIndex > event1Index && ppIndex < event2Index) {
              // LogInfo.logs("PP between :" +ppName);
              if (MarkAndPPClusters.containsKey(ppName))
                return new Pair<String, String>(ppName,
                    Utils.getInverseRelation(MarkAndPPClusters.get(ppName)));
              else
                return new Pair<String, String>(ppName, "");
            } else if (ppIndex > event2Index) {
              // LogInfo.logs("PP after");
            }

            break;
          }
        }
      }
      node = node.parent(root);
    }
    return null;
  }

  private static List<Pair<String, String>> extractMarkRelation(Input input,
      CoreMap sentence1, Tree event1, Tree event2, int trig1) {
    CoreMap sentence = sentence1;
    SemanticGraph graph = sentence
        .get(CollapsedCCProcessedDependenciesAnnotation.class);
    IndexedWord indexedWord1 = Utils.findDependencyNode(sentence, event1), indexedWord2 = Utils
        .findDependencyNode(sentence, event2);
    // ystem.out.println(sentence+", "+event1.toString()+", "+event2.toString());
    // System.out.println("indexed words:" + indexedWord1 + ", "+ indexedWord2);
    List<Pair<String, String>> markRelations = new ArrayList<Pair<String, String>>();
    if (indexedWord1 == null || indexedWord2 == null) {
      return markRelations;
    }
    int event1Index = indexedWord1.index(), event2Index = indexedWord2.index();

    // List<Pair<String, String>> markRelations = new
    // ArrayList<Pair<String,String>>();

    Pair<String, String> markRelation1 = extractMarkRelation(graph, input,
        trig1, indexedWord1, indexedWord2, event1Index, event2Index);
    Pair<String, String> markRelation2 = extractMarkRelation(graph, input,
        trig1, indexedWord2, indexedWord1, event1Index, event2Index);
    if (markRelation1 != null) {
      markRelations.add(new Pair<String, String>(markRelation1.first() + "_1",
          markRelation1.second()));
    }

    if (markRelation2 != null) {
      markRelations.add(new Pair<String, String>(markRelation2.first() + "_2",
          markRelation2.second()));
    }

    return markRelations;
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

  private static Pair<String, String> extractMarkRelation(SemanticGraph graph,
      Input input, int trig1, IndexedWord indexedWordThis,
      IndexedWord indexedWordThat, int event1Index, int event2Index) {
    for (SemanticGraphEdge e : graph.getOutEdgesSorted(indexedWordThis)) {
      if (e.getRelation().getShortName().equals("mark")) {
        int markIndex = e.getTarget().index();
        String markerName = e.getTarget().lemma().toLowerCase();

        // Check if the source of incoming edge (advcl, ccmod, dep) is same as
        // indexedWordThat or parent of indexedWordThat
        if (graph.getIncomingEdgesSorted(indexedWordThis).size() > 0) {
          SemanticGraphEdge edge = graph
              .getIncomingEdgesSorted(indexedWordThis).get(0);

          IndexedWord parent = edge.getSource();

          if (parent.index() == indexedWordThat.index()
              || isInSameDependencyClauseAndChild(graph, parent,
                  indexedWordThat)) {
            if (isFirstEventInSentence(input, trig1) && markIndex < event1Index) {
              // LogInfo.logs("Marker before :" +markerName);
              if (MarkAndPPClusters.containsKey(markerName))
                return new Pair<String, String>(markerName,
                    MarkAndPPClusters.get(markerName));
              else
                return new Pair<String, String>(markerName, "");
            } else if (markIndex < event2Index) {
              // LogInfo.logs("Marker between :" +markerName);
              if (MarkAndPPClusters.containsKey(markerName))
                return new Pair<String, String>(markerName,
                    Utils.getInverseRelation(MarkAndPPClusters.get(markerName)));
              else
                return new Pair<String, String>(markerName, "");
            } else {
              LogInfo.logs("Marker after");
            }
          }
        }
      }
    }
    return null;
  }

  private static boolean isInSameDependencyClauseAndChild(SemanticGraph graph,
      IndexedWord parent, IndexedWord word) {
    List<SemanticGraphEdge> edges = graph.getShortestDirectedPathEdges(parent,
        word);
    if (edges == null)
      return false;

    for (SemanticGraphEdge edge : edges) {
      if (diffClauseRelations.contains(edge.getRelation().getShortName()))
        return false;
    }
    return true;
  }
}
