package edu.stanford.nlp.bioprocess.joint.core;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.stanford.nlp.bioprocess.Utils;

public class Dictionary {
  public static final Set<String> nominalizations = Collections.unmodifiableSet(Utils.getNominalizedVerbs());
  public static final Map<String, String> verbForms = Collections.unmodifiableMap(Utils.getVerbForms());
  public static final Map<String, Integer> clusters = Collections.unmodifiableMap(Utils.loadClustering());  
  public static final List<String> TemporalConnectives = Arrays.asList(new String[] {
      "before", "after", "since", "when", "meanwhile", "lately", "include",
      "includes", "including", "included", "first", "begin", "begins", "began",
      "beginning", "begun", "start", "starts", "started", "starting", "lead",
      "leads", "causes", "cause", "result", "results", "then", "subsequently",
      "previously", "next", "later", "subsequent", "previous" });
  public static final  List<String> diffClauseRelations = Arrays
      .asList(new String[] { "acomp", "advcl", "ccomp", "csubj", "infmod",
          "prepc", "purpcl", "xcomp" });
 
}
