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
import edu.stanford.nlp.parser.lexparser.NoSuchParseException;
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
 * Reads the dataset from a directory with .txt and .ann files (brat files)
 * Allows to get train/dev and cross validation splits
 * TODO - discuss how the labels that I upload work with all the feature extraction (in terms of compatibility)
 * TODO - discuss the correction of spans to entity nodes
 * TODO - fix the creation of candidates in Input
 * TODO - serialize and load
 * @author jonathanberant
 *
 */
public class Dataset {

  public static class Options {
    @Option(gloss = "Paths to read input files (format: <group>:<file>)")
    public ArrayList<Pair<String, String>> inPaths = new ArrayList<Pair<String, String>>();
    @Option(gloss = "Maximum number of examples to read")
    public ArrayList<Pair<String, Integer>> maxExamples = new ArrayList<Pair<String, Integer>>();
    @Option(gloss = "Number of folds for cross validation")
    public int numOfFolds = 2;
    @Option(gloss="verbosity") public int verbose=0;
  }
  public static Options opts = new Options();

  //FIELDS
  private LinkedHashMap<String, List<Pair<Input,Structure>>> allExamples = new LinkedHashMap<String, List<Pair<Input,Structure>>>();
  private StanfordCoreNLP processor;
  private StatFig stats = new StatFig();
  private static Random rand = new Random();

  //METHODS
  public Dataset() {
    Properties props = new Properties();
    props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
    processor = new StanfordCoreNLP(props, false);
  }

  public Set<String> groups() { return allExamples.keySet(); }
  public List<Pair<Input,Structure>> examples(String group) { return allExamples.get(group); }
  public int size(String group) { return allExamples.get(group).size(); }

  public void read() throws IOException, InterruptedException {
    LogInfo.begin_track("Dataset.read");
    readFromPathPairs();
    LogInfo.end_track();
  }

  private void readFromPathPairs() throws IOException, InterruptedException {
    for (Pair<String, String> pathPair : opts.inPaths) {
      String group = pathPair.getFirst();
      String path = pathPair.getSecond();
      List<Pair<Input,Structure>> examples = readFromPath(path,getMaxExamples(group));
      examples = DatasetUtils.shuffle(examples,rand);
      allExamples.put(group, examples);
    }
  }
  private List<Pair<Input, Structure>> readFromPath(String path, int numOfExamples) throws IOException, InterruptedException {
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
    for(String fileName:folder.list(textFilter)){
      File file = new File(folder,fileName);
      LogInfo.begin_track("Reading process %s",new File(folder,fileName));
      Input input = generateInput(file);
      Structure structure = generateStructure(input, 
          new File(folder,fileName.replace(DatasetUtils.TEXT_EXTENSION, DatasetUtils.ANNOTATION_EXTENSION)));
      //res.add(Pair.makePair(input, structure));
      LogInfo.end_track();
    }
    LogInfo.logs(stats);
    return res;
  }

  private Input generateInput(File file) throws IOException, InterruptedException {
    String text = IOUtils.slurpFile(file);
    if(opts.verbose>0)
      LogInfo.logs("Dataset.readFromPath: text=%s",text);
    //get annotation
    Annotation annotation = new Annotation(text);
    //trying 5 times since for some reason the parser throws exception non-deterministically
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
    return new Input(annotation, file.getName().substring(0, file.getName().lastIndexOf('.')));
  }

  private Structure generateStructure(Input input, File file) {

    Map<String,Integer> triggerMap = new HashMap<String,Integer>(); //map from trigger description to token index
    Map<String,IntPair> entityMap = new HashMap<String,IntPair>(); //map from entity description to token span

    //first pass - gather triggers and entity maps
    for(String line: IOUtils.readLines(file)) {
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
    for(String line: IOUtils.readLines(file)) {
      getRolesAndRelations(input, line,triggerMap,entityMap,arguments,relations);
    }

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
        if(DatasetUtils.isRole(edgeLabel)) {
          IntPair span = entityMap.get(id);
          arguments[input.getTriggerIndex(triggerTokenIndex)][input.getArgumentSpanIndex(triggerTokenIndex,span)]=
              DatasetUtils.getLabel(edgeLabel);       
        }
        else if(DatasetUtils.isEventEventRelation(edgeLabel)) {
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
        entityMap.put(tokens[0], DatasetUtils.getSpan(beginToken,endToken+1));
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

  public List<Pair<Input,Structure>> getTrainFold(int foldNum) {
    if(foldNum>=opts.numOfFolds || foldNum < 0) 
      throw new RuntimeException("Illegal fold num, num of folds="+opts.numOfFolds+", fold requested="+foldNum);

    List<Pair<Input,Structure>> res = new ArrayList<Pair<Input,Structure>>();
    List<Pair<Input,Structure>> trainExamples = allExamples.get("train");
    int startIndex = foldNum*(trainExamples.size() / opts.numOfFolds);
    int endIndex = (foldNum+1)*(trainExamples.size() / opts.numOfFolds);

    for(int i = 0; i < startIndex; ++i)
      res.add(trainExamples.get(i));
    for(int i = endIndex; i < trainExamples.size(); ++i)
      res.add(trainExamples.get(i));
    return res;

  }

  public List<Pair<Input,Structure>> getDevFold(int foldNum) {
    if(foldNum>=opts.numOfFolds || foldNum < 0) 
      throw new RuntimeException("Illegal fold num, num of folds="+opts.numOfFolds+", fold requested="+foldNum);

    List<Pair<Input,Structure>> res = new ArrayList<Pair<Input,Structure>>();
    List<Pair<Input,Structure>> trainExamples = allExamples.get("train");
    int startIndex = foldNum*(trainExamples.size() / opts.numOfFolds);
    int endIndex = (foldNum+1)*(trainExamples.size() / opts.numOfFolds);

    for(int i = startIndex; i < endIndex; ++i)
      res.add(trainExamples.get(i));
    return res;
  }

  //STATIC
  private static int getMaxExamples(String group) {
    int maxExamples = Integer.MAX_VALUE;
    for (Pair<String, Integer> maxPair : opts.maxExamples)
      if (maxPair.getFirst().equals(group))
        maxExamples = maxPair.getSecond();
    return maxExamples;
  }

  public static void main(String[] args) throws IOException, InterruptedException {

    opts.inPaths.add(Pair.makePair("train","lib/Dataset/test"));
    Dataset d = new Dataset();
    d.read();
  }
}
