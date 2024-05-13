package ru.ac.phyche.ribyclusters.featuregenerators;

import java.util.HashSet;

import org.apache.commons.lang3.tuple.Pair;

import ru.ac.phyche.ribyclusters.featurepreprocessors.FeaturesPreprocessor;

/**
 * 
 * The feature generator that applies a FeaturesPreprocessor to results of other
 * FeaturesGenerator. The precompute method calls precompute method of
 * underlying FeatureGenerator. Preprocessor should be trained. Usage
 * CombinedFeaturesGenerator, CombinedFeaturesPreprocessor and this method
 * together allows to use any combination of features and preprocessors.
 */
public class PreprocessedFeaturesGenerator extends FeaturesGenerator {

	private FeaturesPreprocessor preproc_ = null;
	private FeaturesGenerator gen_ = null;

	@Override
	public void precompute(HashSet<String> smilesStrings) {
		HashSet<String> smilesStrings1 = new HashSet<String>();
		for (String s : smilesStrings) {
			if (!this.precomputedForMol(s)) {
				smilesStrings1.add(s);
			}
		}
		gen_.precompute(smilesStrings1);
		for (String s : smilesStrings1) {
			float[] f = gen_.featuresForMol(s);
			putPrecomputed(s, preproc_.preprocess(f));
		}
	}

	@Override
	public String getName(int i) {
		return preproc_.featureNames()[i];
	}

	@Override
	public int getNumFeatures() {
		return preproc_.featureNames().length;
	}

	/**
	 * 
	 * @param gen     (non-preprocessed) feature generator. The call of the
	 *                precompute method of PreprocessedFeaturesGenerator causes
	 *                precomputation for the gen instance too.
	 * @param preproc preprocessor (should be preliminary trained!)
	 */
	public PreprocessedFeaturesGenerator(FeaturesGenerator gen, FeaturesPreprocessor preproc) {
		preproc_ = preproc;
		gen_ = gen;
	}

	public Pair<FeaturesGenerator, FeaturesPreprocessor> getGenPreproc() {
		return Pair.of(gen_, preproc_);
	}

}
