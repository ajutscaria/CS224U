package edu.stanford.nlp.bioprocess;

import java.util.Comparator;
import java.util.IdentityHashMap;

import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Pair;

public class PairComparatorByDoubleHashMap implements Comparator<Pair<IdentityHashMap<Tree, String>, Double>> {

    public int compare(Pair<IdentityHashMap<Tree, String>, Double> pr1, Pair<IdentityHashMap<Tree, String>, Double> pr2) {
        return pr2.second.compareTo(pr1.second);
    }
}
