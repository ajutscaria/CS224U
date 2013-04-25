package edu.stanford.nlp.bioprocess.wordcluster;

import java.io.IOException;
import java.io.PrintWriter;
import java.text.Normalizer;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

/**
 * Takes the book-sentences.txt file (one line per sentence with numbering) and outputs a file in the format required by Alex Clark's code for
 * clustering (one word per line, all ASCII).
 * @author jonathanberant
 *
 */
public class ExchangeClusteringPreprocessor {

	public static List<String> Punctuations = Arrays.asList(".", ",","?",":",";","!");
	private StanfordCoreNLP tokenizer;

	public ExchangeClusteringPreprocessor() {
		Properties props = new Properties();
		props.put("annotators", "tokenize");
		tokenizer = new StanfordCoreNLP(props, false);
	}

	/**
	 * @param bookFile - book-sentences file
	 * @throws IOException 
	 */
	public void preprocessBookFile(String bookFile, String preprocessedFile) throws IOException {		
		PrintWriter writer = edu.stanford.nlp.io.IOUtils.getPrintWriter(preprocessedFile);	
		for(String line: IOUtils.readLines(bookFile)) {
			writer.println(preprocessLine(line));
		}	
		writer.close();
	}

	private String preprocessLine(String line) {

		StringBuilder sb = new StringBuilder();
		//the first field is a numbering
		Annotation sentence =  new Annotation(line);
		tokenizer.annotate(sentence);
		boolean first = true;
		for(CoreLabel token: sentence.get(TokensAnnotation.class)) {
			
			//skip first word which is a numberring
			if(first) {
				first = false;
				continue;
			}
			
			String word = preprocessWord(token.get(TextAnnotation.class));
			if(word!=null) {
				sb.append(word+"\n");
			}
		}		
		return sb.toString();
	}

	/**
	 * Delete puncutations and convert to ASCII
	 * @param word
	 * @return preprocessed word
	 */
	private String preprocessWord(String word) {

		if(Punctuations.contains(word))
			return null;
		//convert to ASCII
		String convertedString = Normalizer.normalize(word, Normalizer.Form.NFD).replaceAll("[^\\p{ASCII}]", "");
		return convertedString.toLowerCase();
	}

	public static void main(String[] args) throws IOException {

		ExchangeClusteringPreprocessor ecp = new ExchangeClusteringPreprocessor();
		ecp.preprocessBookFile(args[0],args[1]);
	}


}
