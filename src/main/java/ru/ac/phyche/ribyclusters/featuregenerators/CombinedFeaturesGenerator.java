package ru.ac.phyche.ribyclusters.featuregenerators;

import java.util.HashSet;

/**
 * This feature generator concatenate results of other generators. For example
 * if for a molecule two generators return 1.0, 2.0 and 3.0, 1.0, 5.5
 * respectively the combined generator (created using an array with these
 * generators) will return 1.0, 2.0, 3.0, 1.0, 5.5. The precompute method
 * invokes the precompute method of each of generators. There are no need to
 * call them explicitly.
 *
 */
public class CombinedFeaturesGenerator extends FeaturesGenerator {

	private FeaturesGenerator[] generators_ = null;
	private String[] names = null;
	private int n = 0;

	@Override
	public void precompute(HashSet<String> smilesStrings) {
		HashSet<String> smilesStrings1 = new HashSet<String>();
		for (String s : smilesStrings) {
			if (!this.precomputedForMol(s)) {
				smilesStrings1.add(s);
			}
		}
		for (int i = 0; i < generators_.length; i++) {
			generators_[i].precompute(smilesStrings1);
		}
		for (String st : smilesStrings1) {
			float[] features = new float[getNumFeatures()];
			int i = 0;
			for (int j = 0; j < generators_.length; j++) {
				float[] features_j = generators_[j].featuresForMol(st);
				for (int k = 0; k < features_j.length; k++) {
					features[i] = features_j[k];
					i++;
				}
			}
			if (i != features.length) {
				throw (new RuntimeException("Wrong number of features"));
			}
			this.putPrecomputed(st, features);
		}
	}

	@Override
	public String getName(int i) {
		return names[i];
	}

	@Override
	public int getNumFeatures() {
		return n;
	}

	@SuppressWarnings("unused")
	private CombinedFeaturesGenerator() {

	}

	/**
	 * 
	 * @param generators other feature generators those will be used. Generators
	 *                   should be initialized (and non-null), but can handle no
	 *                   precomputed features.
	 */
	public CombinedFeaturesGenerator(FeaturesGenerator[] generators) {
		generators_ = generators;
		int i = 0;
		for (int j = 0; j < generators_.length; j++) {
			n += generators_[j].getNumFeatures();
		}
		names = new String[n];
		for (int j = 0; j < generators_.length; j++) {
			for (int k = 0; k < generators_[j].getNumFeatures(); k++) {
				names[i] = generators_[j].getName(k);
				i++;
			}
		}
		if (i != names.length) {
			throw (new RuntimeException("Wrong number of features"));
		}
	}
}
