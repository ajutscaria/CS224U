package edu.stanford.nlp.bioprocess;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ie.machinereading.domains.bionlp.BioNLPFormatReader;
import edu.stanford.nlp.ie.machinereading.msteventextractor.GENIA09DataSet;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.Pair;

public class Dataset {
  final String ANNOTATED_FILE_EXTENSION = ".ann";
  LinkedHashMap<String, List<Example>> allExamples;
  LinkedHashMap<String, String> inPaths;
  ArrayList<Pair<String, Integer>> maxExamples;
  
  public Dataset() {
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
    //System.out.println(allExamples.size());
  }
  /*
  private List<Example> readExamples(String folderName){
    List<Example> examples = new ArrayList<Example>();
    try {
      BufferedReader reader = new BufferedReader(new FileReader(fileName));
      String line;
      while((line = reader.readLine())!=null) {
        Example ex = new Example();
        String[] splits = line.split("\t");
        ex.id = splits[0];
        ex.data = splits[1];
        ex.gold = readAnnotation(folderName, ex.id + ANNOTATED_FILE_EXTENSION);
        
      }
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    return examples;
  }*/
  
  /*
  private ArrayList<Event> readAnnotation(String folderName, String fileName){
    ArrayList<Event> events= new ArrayList<Event>();
    HashMap<String, Event> E = new HashMap<String, Event>();
    try {
      BufferedReader reader = new BufferedReader(new FileReader(folderName + fileName));
      String line;
      HashMap<String, String> T = new HashMap<String, String>();
      while((line = reader.readLine())!=null) {
        String[] splits = line.split("\t");
        String desc = splits[0];
        switch(desc.charAt(0)) {
          case 'T':
            String type = splits[1].split(" ")[0];
            T.put(desc, splits[2]);
            System.out.println(desc + ":" + splits[2]);
            break;
          case 'E':
            for(String spl:splits[1].split(" ")){
              String property = spl.split(":")[0], value = spl.split(":")[1];
              System.out.println(desc + " : " + property + " = "+value);
            }
            break;
          case '*':
            break;
        }
      }
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    return events;
  }
  */
}
