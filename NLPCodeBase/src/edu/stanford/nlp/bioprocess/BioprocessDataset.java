package edu.stanford.nlp.bioprocess;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.Pair;

public class BioprocessDataset {
  final String ANNOTATED_FILE_EXTENSION = ".ann";
  LinkedHashMap<String, List<Example>> allExamples;
  LinkedHashMap<String, String> inPaths;
  ArrayList<Pair<String, Integer>> maxExamples;
  
  public BioprocessDataset() {
    allExamples = new LinkedHashMap<String, List<Example>>();
    inPaths = new LinkedHashMap<String, String>();
    maxExamples = new ArrayList<Pair<String,Integer>>();
  }
  
  public void addGroup(String groupName, String path) {
    inPaths.put(groupName, path);
  }
  
  public Set<String> groups() {
    return allExamples.keySet();
  }
  
  public List<Example> examples(String group) {
    return allExamples.get(group);
  }
  
  public void read() {
    Properties props = new Properties();
    props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");

    StanfordCoreNLP processor = new StanfordCoreNLP(props, false);
    BioProcessFormatReader reader = new BioProcessFormatReader();
    reader.setProcessor(processor);

    for(String group:inPaths.keySet()) {
      String folderName = inPaths.get(group);
      try {
        allExamples.put(group, reader.parseFolder(folderName));
      } catch (IOException e) {
        // TODO Auto-generated catch block
        System.out.println("Exception - " + e.getMessage());
        e.printStackTrace();
      }
    }
  }
}
