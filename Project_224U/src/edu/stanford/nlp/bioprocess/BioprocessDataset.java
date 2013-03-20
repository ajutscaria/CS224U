package edu.stanford.nlp.bioprocess;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.Pair;
import fig.basic.LogInfo;

/***
 * Dataset used to store the bio-process related data samples.
 * @author Aju
 *
 */

public class BioprocessDataset {
	final String ANNOTATED_FILE_EXTENSION = ".ann";
	//Map between the type of data (e.g. train or test) and the list of examples of that type.
	LinkedHashMap<String, List<Example>> allExamples;
	//Map between the type of data (e.g. train or test) and the input folder to read them.
	LinkedHashMap<String, String> inPaths;
	//Maximum number of examples for a type of data - Unused as of now.
	ArrayList<Pair<String, Integer>> maxExamples;
	BioProcessFormatReader reader;
	StanfordCoreNLP processor;

	/***
	 * Initialize the dataset with the types of data and its input folders.
	 * @param inPaths
	 */
	public BioprocessDataset(HashMap<String, String> inPaths) {
		allExamples = new LinkedHashMap<String, List<Example>>();
		this.inPaths = new LinkedHashMap<String, String>();
		for(String str:inPaths.keySet())
			addGroup(str, inPaths.get(str));
		maxExamples = new ArrayList<Pair<String,Integer>>();
		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");

		processor = new StanfordCoreNLP(props, false);
		reader = new BioProcessFormatReader();
		reader.setProcessor(processor);
	}

	/***
	 * Add a type of data and its input path to the data set.
	 * @param groupName
	 * @param path
	 */
	public void addGroup(String groupName, String path) {
		inPaths.put(groupName, path);
	}

	/***
	 * Get all the types of data in the dataset.
	 * @return
	 */
	public Set<String> groups() {
		return allExamples.keySet();
	}

	/***
	 * Get the types of examples belonging to a type of data.
	 * @param group
	 * @return
	 */
	public List<Example> examples(String group) {
		return allExamples.get(group);
	}

	/***
	 * Read all types of data and load it to the dataset.
	 */
	public void readAll() {
		for(String group:inPaths.keySet()) {
			read(group);
		}
	}

	public void read(String group) {

		LogInfo.begin_track("Reading data");
		String folderName = inPaths.get(group);
		try {
			allExamples.put(group, reader.parseFolder(folderName));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			LogInfo.logs("Exception - " + e.getMessage());
			e.printStackTrace();
		}
		LogInfo.end_track();
	}
}
