package ru.ac.phyche.ribyclusters.featurepreprocessors;

import java.util.ArrayList;

import ru.ac.phyche.ribyclusters.ChemDataset;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;

/**
 * 
 * This preprocessor drops features that contain NaN values in the training set.
 * If maxNaNsFraction == 0 - any NaN value will cause exclusion of entire
 * feature. maxNaNsFraction - max fraction of NaN values
 *
 */
public class DropFeaturesWithNaNsPreprocessor extends DropFeaturesPreprocessor {

	float maxNaNsFraction = 0;

	@Override
	public void train(FeaturesGenerator features, ChemDataset data) {
		features.precompute(data);
		float[][] featuresFloat = features.features(data);
		ArrayList<Integer> drop = new ArrayList<Integer>();
		ArrayList<String> namesRetain = new ArrayList<String>();
		if (featuresFloat.length * features.getNumFeatures() != 0) {
			for (int i = 0; i < features.getNumFeatures(); i++) {
				int nNaNs = 0;
				for (int j = 0; j < featuresFloat.length; j++) {
					if (Float.isNaN(featuresFloat[j][i])) {
						nNaNs++;
					}

				}
				if ((((float) nNaNs) / ((float) featuresFloat.length)) > maxNaNsFraction) {
					drop.add(i);
				} else {
					namesRetain.add(features.getName(i));
				}
			}
			int[] result = new int[drop.size()];
			for (int i = 0; i < result.length; i++) {
				result[i] = drop.get(i);
			}
			setFeaturesToDrop(result);
			setNames(namesRetain.toArray(new String[namesRetain.size()]));
		} else {
			setNames(features.getNames());
			setFeaturesToDrop(new int[] {});
		}
	}

	/**
	 * maxNaNsFraction = 0
	 */
	public DropFeaturesWithNaNsPreprocessor() {
		maxNaNsFraction = 0;
	}

	/**
	 * 
	 * @param maxNaNsFraction allowed fraction of NaN values. Allows to retain
	 *                        features with low non-zero number of NaNs (for example
	 *                        1-2 per train set with 1000 compounds).
	 */
	public DropFeaturesWithNaNsPreprocessor(float maxNaNsFraction) {
		this.maxNaNsFraction = maxNaNsFraction;
	}
}
