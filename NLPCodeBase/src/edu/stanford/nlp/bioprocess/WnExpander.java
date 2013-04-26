package edu.stanford.nlp.bioprocess;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.util.OneToOneMap.OneToOneMapException;
import edu.stanford.nlp.wsd.WordNet;
import edu.stanford.nlp.wsd.WordNet.EdgeType;
import edu.stanford.nlp.wsd.WordNet.WordID;
import edu.stanford.nlp.wsd.WordNet.WordNetID;
import fig.basic.LogInfo;
import fig.basic.Option;

public class WnExpander {
  
  public static class Options {
    @Option(gloss="Verbose") public int verbose = 1;
    @Option(gloss="Path to Wordnet file") public String wnFile="lib/wordnet-3.0-prolog";
    @Option(gloss="Relations to expand with wordnet") public Set<String> wnRelations = new HashSet<String>(); 
  }
  public static Options opts = new Options();

  private WordNet wn;
  private Set<EdgeType> edgeTypes = new HashSet<WordNet.EdgeType>();
  
  /**
   * Initializing wordnet and the relations to expand with
   * @throws IOException
   * @throws OneToOneMapException 
   */
  public WnExpander() throws IOException, OneToOneMapException {
    wn = WordNet.loadPrologWordNet(new File(opts.wnFile));
    for(String wnRelation: opts.wnRelations) {
      if(wnRelation.equals("derives"))
        edgeTypes.add(EdgeType.DERIVES);
      else if(wnRelation.equals("derived_from"))
        edgeTypes.add(EdgeType.DERIVED_FROM);
      else if(wnRelation.equals("hyponym"))
        edgeTypes.add(EdgeType.HYPONYM);
    }
  }
  
  public Set<String> expandPhrase(String phrase) {

    //find synsetse for phrase
    Set<WordNetID> phraseSynsets = phraseToSynsets(phrase);
    //expand synsets
    for(EdgeType edgeType: edgeTypes)
      phraseSynsets.addAll(expandSynsets(phraseSynsets, edgeType));
    //find phrases for synsets
    Set<String> expansions = synsetsToPhrases(phraseSynsets);
    if(opts.verbose>0) {
      for(String expansion: expansions)
      LogInfo.logs("WordNetExpansionLexicon: expanding %s to %s",phrase,expansion);
    }
    return expansions;
  }
  
  public Set<String> getSynonyms(String phrase) {
    Set<WordNetID> phraseSynsets = phraseToSynsets(phrase);
    Set<String> expansions = synsetsToPhrases(phraseSynsets);
    expansions.remove(phrase);
    return expansions;
  }
  
  public Set<String> getDerivations(String phrase) {
    Set<WordNetID> phraseSynsets = phraseToSynsets(phrase);
    Set<WordNetID> derivations = new HashSet<WordNet.WordNetID>();
    derivations.addAll(expandSynsets(phraseSynsets,EdgeType.DERIVED_FROM));
    derivations.addAll(expandSynsets(phraseSynsets,EdgeType.DERIVES));
    Set<String> expansions = synsetsToPhrases(derivations);
    expansions.remove(phrase);
    return expansions;
  }
  
  public Set<String> getHypernyms(String phrase) {
    Set<WordNetID> phraseSynsets = phraseToSynsets(phrase);
    Set<WordNetID> hypernyms = new HashSet<WordNet.WordNetID>();
    hypernyms.addAll(expandSynsets(phraseSynsets,EdgeType.HYPONYM));
    Set<String> expansions = synsetsToPhrases(hypernyms);
    expansions.remove(phrase);
    return expansions;
  }
  
  public Set<WordNetID> getSynsets(String word, String posType) {
	  char type = posType.startsWith("VB") ? 'v' : posType.startsWith("NN") ? 'n' : 'o';
	  if(type == 'o')
		  return null;
	  Set<WordNetID> wordTags = new HashSet<WordNet.WordNetID>();
	  WordID wordID = wn.getWordID(word);
	  if(wordID==null)
		  return null;
	  wordTags.addAll(wordID.get(EdgeType.WORD_TO_WORDTAG));
	  Set<WordNetID> synsets = new HashSet<WordNet.WordNetID>();
	  for(WordNetID wordTag: wordTags) {
		  if(wordTag.toString().endsWith("#" + type))
			  synsets.addAll(wordTag.get(EdgeType.WORDTAG_IN_SYNSET));
	    /*System.out.println(wordTag + " : " + wordTag.get(EdgeType.WORDTAG_IN_SYNSET));
	    for(WordNetID synSet: synsets) {
	  	  System.out.println("\tSynset " + synSet);
	  	  for(WordNetID words: synSet.get(EdgeType.SYNSET_HAS_WORDTAG))
	  		  System.out.println("\t\tWord " + words);
	    }*/
	  }
	  return synsets;    
	}

  private Set<String> synsetsToPhrases(Set<WordNetID> phraseSynsets) {

    Set<String> res = new HashSet<String>();
    for(WordNetID phraseSynset: phraseSynsets) {
      res.addAll(synsetToPhrases(phraseSynset));
    }
    return res;
  }

  private Collection<String> synsetToPhrases(WordNetID phraseSynset) {
    Set<String> res = new HashSet<String>();
    List<WordNetID> wordTags = phraseSynset.get(EdgeType.SYNSET_HAS_WORDTAG);
    for(WordNetID wordTag: wordTags)  {
      List<WordNetID> words = wordTag.get(EdgeType.WORDTAG_TO_WORD);
      for(WordNetID word: words) {
        res.add(((WordID)word).word);
      }
    }
    return res;
  }

  /**
   * Given a phrase find all synsets containing this phrase
   * @param phrase
   * @return
   */
  private Set<WordNetID> phraseToSynsets(String phrase) {

    List<WordNetID> wordTags = new LinkedList<WordNet.WordNetID>();
    WordID word = wn.getWordID(phrase);
    if(word!=null)
      wordTags.addAll(word.get(EdgeType.WORD_TO_WORDTAG));
    Set<WordNetID> synsets = new HashSet<WordNet.WordNetID>();
    for(WordNetID wordTag: wordTags) {
      synsets.addAll(wordTag.get(EdgeType.WORDTAG_IN_SYNSET));
    }
    return synsets;    
  }

  private List<WordNetID> expandSynset(WordNetID synset, EdgeType edgeType) {
    return synset.get(edgeType);
  }

  private Set<WordNetID> expandSynsets(Collection<WordNetID> synsets, EdgeType edgeType) {
    Set<WordNetID> res = new HashSet<WordNet.WordNetID>();
    for(WordNetID synset: synsets)
      res.addAll(expandSynset(synset, edgeType));
    return res;
  }
  
  public static void main(String[] args) throws IOException, OneToOneMapException {

    WnExpander wnLexicon = new WnExpander();
    System.out.println(wnLexicon.getSynsets("attack", "VBZ"));
    System.out.println();
  }

}
