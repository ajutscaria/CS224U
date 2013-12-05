package edu.stanford.nlp.bioprocess.joint.reader;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;

import edu.stanford.nlp.bioprocess.joint.core.Input;
import edu.stanford.nlp.bioprocess.joint.core.Structure;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.Pair;
import fig.basic.StatFig;

/**
 * Reads the dataset from .txt and .ann files (brat files)
 * Allows to get train/dev and cross validation splits
 * TODO - discuss how the labels that I upload work with all the feature extraction (in terms of compaibility)
 * TODO - discuss the correction of spans to entity nodes
 * TODO - discuss the methods needed from Input
 * @author jonathanberant
 *
 */
public class Dataset {

  public static class Options {
    @Option(gloss = "Paths to read input files (format: <group>:<file>)")
    public ArrayList<Pair<String, String>> inPaths = new ArrayList<Pair<String, String>>();
    @Option(gloss = "Maximum number of examples to read")
    public ArrayList<Pair<String, Integer>> maxExamples = new ArrayList<Pair<String, Integer>>();

    // Training file gets split into:
    // |  trainFrac  -->  |           | <-- devFrac |
    @Option(gloss = "Fraction of trainExamples (from the beginning) to keep for training")
    public double trainFrac = 1;
    @Option(gloss = "Fraction of trainExamples (from the end) to keep for development")
    public double devFrac = 0;
    @Option(gloss = "Used to randomly divide training examples")
    public Random splitRandom = new Random(1);
    @Option(gloss="verbosity") public int verbose=0;
  }
  public static Options opts = new Options();

  //FIELDS
  private LinkedHashMap<String, List<Pair<Input,Structure>>> allExamples = new LinkedHashMap<String, List<Pair<Input,Structure>>>();
  private StanfordCoreNLP processor;
  private StatFig stats = new StatFig();

  //METHODS
  public Dataset() {
    Properties props = new Properties();
    props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
    processor = new StanfordCoreNLP(props, false);
  }

  public Set<String> groups() { return allExamples.keySet(); }
  public List<Pair<Input,Structure>> examples(String group) { return allExamples.get(group); }
  public int size(String group) { return allExamples.get(group).size(); }

  public void read() throws IOException {
    LogInfo.begin_track("Dataset.read");
    readFromPathPairs();
    LogInfo.end_track();
  }

  private void readFromPathPairs() throws IOException {
    for (Pair<String, String> pathPair : opts.inPaths) {
      String group = pathPair.getFirst();
      String path = pathPair.getSecond();
      allExamples.put(group, readFromPath(path,getMaxExamples(group)));
    }    
  }
  private List<Pair<Input, Structure>> readFromPath(String path, int numOfExamples) throws IOException {
    List<Pair<Input,Structure>> res = new ArrayList<Pair<Input,Structure>>();

    File folder = new File(path);
    FilenameFilter textFilter = new FilenameFilter() {
      public boolean accept(File dir, String name) {
        String lowercaseName = name.toLowerCase();
        if (lowercaseName.endsWith(DatasetUtils.TEXT_EXTENSION)) 
          return true;
        else 
          return false;
      }
    };
    //read examples
    for(String file:folder.list(textFilter)){
      LogInfo.begin_track("Reading process %s",file);
      Input input = generateInput(file);
      Structure structure = generateStructure(input);
      res.add(Pair.makePair(input, structure));
    }
    return res;
  }

  private Input generateInput(String file) throws IOException {
    String text = IOUtils.slurpFile(file);
    if(opts.verbose>0)
      LogInfo.logs("Dataset.readFromPath: text=%s",text);
    //get annotation
    Annotation annotation = new Annotation(text);
    processor.annotate(annotation);
    //get some stats
    List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
    stats.add("files",1.0);
    stats.add("sentences",sentences.size());

    for(CoreMap sentence:sentences) {
      stats.add("tokens",sentence.get(TokensAnnotation.class).size());
      Tree syntacticParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
      syntacticParse.setSpans();
    }
    return new Input(annotation, file.substring(0, file.lastIndexOf('.')));
  }

  private Structure generateStructure(Input input) {

    Map<String,Integer> triggerMap = new HashMap<String,Integer>(); //map from trigger description to token index
    Map<String,IntPair> entityMap = new HashMap<String,IntPair>(); //map from entity description to token span

    //first pass - gather triggers and entity maps
    for(String line: IOUtils.readLines(input.id+DatasetUtils.ANNOTATION_EXTENSION)) {
      getTriggersAndEntities(input,line,triggerMap,entityMap);
    }
    //generate trigger array
    String[] triggers = generateTriggersFromMap(input,triggerMap);
    //init relations and argument arrays
    String[] relations = new String[input.getNumberOfEERelationCandidates()];
    String[][] arguments = new String[input.getNumberOfTriggers()][];
    for(int i = 0; i < arguments.length; ++i) 
      arguments[i] = new String[input.getNumberOfArgumentCandidates(i)];
    
    //second pass - populate relations and arguments
    for(String line: IOUtils.readLines(input.id+DatasetUtils.ANNOTATION_EXTENSION)) {
      getRolesAndRelations(input, line,triggerMap,entityMap,arguments,relations);
    }

    LogInfo.logs(stats);
    Structure res = new Structure(input, triggers, arguments, relations);
    return res;
  }

  private String[] generateTriggersFromMap(Input input, Map<String, Integer> triggerMap) {
    String[] res = new String[input.getNumberOfTriggers()];
    Arrays.fill(res, DatasetUtils.OTHER_LABEL);
    for(Integer tokenIndex: triggerMap.values()) {
      res[input.getTriggerIndex(tokenIndex)] = DatasetUtils.EVENT_LABEL;
    }
    return res;
  }

  private void getRolesAndRelations(Input input, String line,
      Map<String, Integer> triggerMap, Map<String, IntPair> entityMap, String[][] arguments, String[] relations) {

    if(line.startsWith("E")) {
      String[] parts = line.split("\t");
      String[] eventDetails = parts[1].split("\\s+");
      assert DatasetUtils.isEvent(eventDetails[1].split(":")[0]);

      int triggerTokenIndex = triggerMap.get(eventDetails[0].split(":")[1]);
      for(int i = 1; i < eventDetails.length; ++i) {
        String[] dependentParts = eventDetails[i].split(":");
        String edgeLabel = dependentParts[0];
        String id = dependentParts[1];
        if(DatasetUtils.isEventEventRelation(edgeLabel)) {
          IntPair span = entityMap.get(id);
          arguments[input.getTriggerIndex(triggerTokenIndex)][input.getArgumentSpanIndex(span)]=
              DatasetUtils.getLabel(edgeLabel);       
        }
        else if(DatasetUtils.isRole(edgeLabel)) {
          int otherTriggerTokenIndex = triggerMap.get(id);
          int triggerId1 = input.getTriggerIndex(triggerTokenIndex);
          int triggerId2 = input.getTriggerIndex(otherTriggerTokenIndex);
          relations[input.getEERelationIndex(triggerId1,triggerId2)]=DatasetUtils.getLabel(edgeLabel);
        }
        else throw new RuntimeException("Line contains unknown event-event relation or role: " + line);
      }
    }  
  }

  private void getTriggersAndEntities(Input input, String line,
      Map<String, Integer> triggerMap, Map<String, IntPair> entityMap) {

    int[] beginOffset = DatasetUtils.mapCharBeginOffsetToTokenIndex(input.annotation.get(TokensAnnotation.class));
    int[] endOffset = DatasetUtils.mapCharEndOffsetToTokenIndex(input.annotation.get(TokensAnnotation.class));
    String[] tokens = line.split("\t");
    if(tokens[0].charAt(0)=='T') {
      String[] offsets = tokens[1].split("\\s+");
      assert offsets.length==3;

      int beginIndex = Integer.parseInt(offsets[1]);
      int endIndex = Integer.parseInt(offsets[2]);
      int beginToken = beginOffset[beginIndex];
      int endToken = endOffset[endIndex];
      assert beginToken != -1 && endToken != -1;

      if(DatasetUtils.isEvent(offsets[0])) {
        assert beginToken==endToken; // events are single words
        triggerMap.put(tokens[0], beginToken);
      }
      else if(offsets[0].equals(DatasetUtils.ENTITY_TYPE)) {
        entityMap.put(offsets[0], DatasetUtils.getSpan(beginToken,endToken+1));
      }
      else throw new RuntimeException("Line does not specify a trigger or entity: " + line);
    }
    else if (tokens[0].charAt(0)=='E') {
      String[] info = tokens[1].split("\\s+");
      String[] eventId = info[0].split(":");
      assert DatasetUtils.isEvent(eventId[0]);
      assert triggerMap.containsKey(eventId[1]);
      triggerMap.put(tokens[0], triggerMap.get(eventId[1]));
    }
  }

  //STATIC
  private static int getMaxExamples(String group) {
    int maxExamples = Integer.MAX_VALUE;
    for (Pair<String, Integer> maxPair : opts.maxExamples)
      if (maxPair.getFirst().equals(group))
        maxExamples = maxPair.getSecond();
    return maxExamples;
  }
}
