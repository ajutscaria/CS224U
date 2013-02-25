package edu.stanford.nlp.bioprocess;

import java.util.*;

public class Viterbi {

	private final Index labelIndex;
	private final Index featureIndex;
	private final double[][] weights;

	public Viterbi(Index labelIndex, Index featureIndex, double[][] weights) {
		this.labelIndex = labelIndex;
		this.featureIndex = featureIndex;
		this.weights = weights;
	}

	public void decode(List<Datum> data, List<Datum> dataWithMultiplePrevLabels) {
		// load words from the data
		List<String> words = new ArrayList<String>();
		for (Datum datum : data) {
			words.add(datum.word);
		}

		// for each position in data
		for (int position = 0; position< data.size(); position++) {
			int bestLabelIndex = -1;
			double bestScore = Integer.MIN_VALUE;
			for(int labelCounter = 0; labelCounter < labelIndex.size(); labelCounter++) {
				double score = computeScore(dataWithMultiplePrevLabels.get(position * labelIndex.size() + labelCounter).features, labelCounter);
				if(score > bestScore) {
					bestScore = score;
					bestLabelIndex = labelCounter;
				}
			}
			data.get(position).guessLabel = labelIndex.get(bestLabelIndex).toString();
		}
	}

	private double computeScore(List<String> features, int labelNumber) {
		double score = 0;

		for (Object feature : features) {
			int f = featureIndex.indexOf(feature);
			if (f < 0) {
				continue;
			}
			score += weights[labelNumber][f];
		}

		return score;
	}

	private int numLabels() {
		return labelIndex.size();
	}

}