package edu.stanford.nlp.bioprocess;

import java.util.Comparator;

import edu.stanford.nlp.util.Pair;

public class PairComparatorByDouble implements Comparator<Pair<String, Double>> {

    public int compare(Pair<String, Double> pr1, Pair<String, Double> pr2) {
        //return pr2.second.compareTo(pr1.second);
        if (pr2.second > pr1.second) {
        	return 1;
        } else {
        	return -1;
        }
    }
}
