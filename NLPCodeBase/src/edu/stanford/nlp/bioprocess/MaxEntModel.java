package edu.stanford.nlp.bioprocess;

import java.util.*;

public class MaxEntModel {

	private final Index labelIndex;
	private final Index featureIndex;
	private final double[][] weights;

	public MaxEntModel(Index labelIndex, Index featureIndex, double[][] weights) {
		this.labelIndex = labelIndex;
		this.featureIndex = featureIndex;
		this.weights = weights;
	}

	public void decodeForEntity(List<Datum> data, List<Datum> dataWithMultiplePrevLabels) {
		// for each position in data
		for (int position = 0; position< data.size(); position++) {
			double scoreO = computeScore(dataWithMultiplePrevLabels.get(position * labelIndex.size() + 0).features, 0);
			double scoreE = computeScore(dataWithMultiplePrevLabels.get(position * labelIndex.size() + 1).features, 1);

			int bestLabelIndex = scoreO > scoreE ? 0 : 1;
			data.get(position).guessLabel = labelIndex.get(bestLabelIndex).toString();
			data.get(position).setProbability(Math.exp(scoreE) / (Math.exp(scoreE) + Math.exp(scoreO)));
		}
	}

	private double computeScore(FeatureVector features, int labelNumber) {
		double score = 0;

		for (Object feature : features.getFeatures()) {
			int f = featureIndex.indexOf(feature);
			if (f < 0) {
				continue;
			}
			score += weights[labelNumber][f];
		}

		return score;
	}
}