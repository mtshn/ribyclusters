package ru.ac.phyche.ribyclusters.featurepreprocessors;

import java.util.ArrayList;

import ru.ac.phyche.ribyclusters.ChemDataset;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;

/**
 * 
 * This preprocessor drops features that are constant (all values are equal,
 * zero variation) in the training set. See also docs for superclass.
 *
 */
public class DropConstantFeaturesPreprocessor extends DropFeaturesPreprocessor {

	@Override
	public void train(FeaturesGenerator features, ChemDataset data) {
		features.precompute(data);
		float[][] featuresFloat = features.features(data);
		ArrayList<Integer> drop = new ArrayList<Integer>();
		ArrayList<String> namesRetain = new ArrayList<String>();
		if (featuresFloat.length * features.getNumFeatures() != 0) {
			for (int i = 0; i < features.getNumFeatures(); i++) {
				boolean allIdentical = true;
				float all = featuresFloat[0][i];
				for (int j = 0; j < featuresFloat.length; j++) {
					if (featuresFloat[j][i] != all) {
						allIdentical = false;
					}

				}
				if (allIdentical) {
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
}
