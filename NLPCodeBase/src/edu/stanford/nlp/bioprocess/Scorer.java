package edu.stanford.nlp.bioprocess;

import java.util.*;

import edu.stanford.nlp.util.Triple;

public class Scorer {

  public static Triple<Double, Double, Double> score(List<Datum> data) {
    int tp = 0, fp = 0, fn = 0;

    for (Datum d:data) {
    	if(d.label.equals("E")) {
    		if(d.guessLabel.equals("E"))
    			tp++;
    		else
    			fn++;
    	}
    	if(d.label.equals("O")) {
    		if(d.guessLabel.equals("E"))
    			fp++;
    	}
    }
    double precision = (double)tp/(tp+fp), recall = (double)tp/(tp+fn);
    double f= 2 * precision * recall / (precision + recall);
    
    //System.out.println("precision = "+precision);
    //System.out.println("recall = "+recall);
    //System.out.println("F1 = "+f);
    
    return new Triple<Double, Double, Double>(precision, recall, f);
  }
}