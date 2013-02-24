package edu.stanford.nlp.bioprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import edu.stanford.nlp.bioprocess.BioProcessAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;

public class FeatureFactory {

    /** Add any necessary initialization steps for your features here.
     *  Using this constructor is optional. Depending on your
     *  features, you may not need to intialize anything.
     */
    public FeatureFactory() {

    }

    /**
     * Words is a list of the words in the entire corpus, previousLabel is the label
     * for position-1 (or O if it's the start of a new sentence), and position
     * is the word you are adding features for. PreviousLabel must be the
     * only label that is visible to this method. 
     */
    private List<String> computeFeatures(CoreMap sentence, CoreLabel entity) {

	List<String> features = new ArrayList<String>();

	String currentWord = entity.originalText();

	// Baseline Features  
	if(entity.index() > 0) {
		CoreLabel prev = sentence.get(TokensAnnotation.class).get(entity.index() - 1);
		features.add("prevword=" + prev.originalText());
		features.add("prevpos=" + prev.get(PartOfSpeechAnnotation.class));
	}
	features.add("word=" + currentWord);
	features.add("pos=" + entity.get(PartOfSpeechAnnotation.class));
	//features.add("trueCase=" + entity.get(TrueCaseAnnotation.class));
	features.add("lemma=" + entity.get(LemmaAnnotation.class));
	//features.add("ner=" + entity.get(NamedEntityTagAnnotation.class));
	//features.add("role=" + entity.get(RoleAnnotation.class));
	features.add("stem=" + entity.get(StemAnnotation.class));
	//features.add("prevLabel=" + previousLabel);
	//features.add("word=" + currentWord + ", prevLabel=" + previousLabel);
	/** Warning: If you encounter "line search failure" error when
	 *  running the program, considering putting the baseline features
	 *  back. It occurs when the features are too sparse. Once you have
	 *  added enough features, take out the features that you don't need. 
	 */

	return features;
    }


    /** Do not modify this method **/
    public List<Datum> readData(String filename) throws IOException {

	List<Datum> data = new ArrayList<Datum>();
	BufferedReader in = new BufferedReader(new FileReader(filename));

	for (String line = in.readLine(); line != null; line = in.readLine()) {
	    if (line.trim().length() == 0) {
		continue;
	    }
	    String[] bits = line.split("\\s+");
	    String word = bits[0];
	    String label = bits[1];

	    Datum datum = new Datum(word, label);
	    data.add(datum);
	}

	return data;
    }

    /** Do not modify this method **/
    public List<Datum> readTestData(String ch_aux) throws IOException {

	List<Datum> data = new ArrayList<Datum>();

	for (String line : ch_aux.split("\n")) {
	    if (line.trim().length() == 0) {
		continue;
	    }
	    String[] bits = line.split("\\s+");
	    String word = bits[0];
	    String label = bits[1];

	    Datum datum = new Datum(word, label);
	    data.add(datum);
	}

	return data;
    }

    /** Do not modify this method **/
    public List<Datum> setFeaturesTrain(List<Example> data) {
	List<Datum> newData = new ArrayList<Datum>();
	
	for (Example ex : data) {
		List<CoreLabel> entityHeads = getEntityTokens(ex);
		for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
			String prevLabel = "O";
			for(CoreLabel label: sentence.get(TokensAnnotation.class)) {
				String type = "O";
				if(entityHeads.contains(label)) {
					type = "E";
				}
				Datum newDatum = new Datum(label.originalText(), type);
				newDatum.features = computeFeatures(sentence, label);
				newDatum.previousLabel = prevLabel;
				newData.add(newDatum);
				
				prevLabel = newDatum.previousLabel;
			}
		}
	}

	return newData;
    }
    
    /** Do not modify this method **/
    public List<Datum> setFeaturesTest(List<Example> data) {
	// this is so that the feature factory code doesn't accidentally use the
	// true label info
	List<Datum> newData = new ArrayList<Datum>();
	List<String> labels = new ArrayList<String>();
	Map<String, Integer> labelIndex = new HashMap<String, Integer>();

	labelIndex.put("O", 0);
	labelIndex.put("E", 1);
	labels.add("O");
	labels.add("E");

	// compute features for all possible previous labels in advance for
	// Viterbi algorithm
	String prevLabel = "O";
	int i = 0;
	for (Example ex : data) {
		List<CoreLabel> entityHeads = getEntityTokens(ex);
		for(CoreMap sentence : ex.gold.get(SentencesAnnotation.class)) {
			for(CoreLabel token: sentence.get(TokensAnnotation.class)) {
				if (i == 0) {
					String type = "O";
					if(entityHeads.contains(token)) {
						type = "E";
					}
					Datum newDatum = new Datum(token.originalText(), type);
					newDatum.features = computeFeatures(sentence, token);
					newDatum.previousLabel = prevLabel;
					newData.add(newDatum);

				} 
				else {
					for (String previousLabel : labels) {
						String type = "O";
						if(entityHeads.contains(token)) {
							type = "E";
							}
						
						Datum newDatum = new Datum(token.originalText(), type);
						newDatum.features = computeFeatures(sentence, token);
						newDatum.previousLabel = previousLabel;
						newData.add(newDatum);
					}
			    }
				i++;
			}
		}

	}

	return newData;
	
    }

    private List<CoreLabel> getEntityTokens(Example ex) {
    	List<CoreLabel> lst = new ArrayList<CoreLabel>();
    	for(EntityMention entity : ex.gold.get(EntityMentionsAnnotation.class))
    		lst.add(entity.getHeadToken());
    	return lst;
    }
}
