package ru.ac.phyche.ribyclusters.featurepreprocessors;

import java.util.ArrayList;
import java.util.Arrays;

import ru.ac.phyche.ribyclusters.ArUtls;
import ru.ac.phyche.ribyclusters.ChemDataset;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;

// Strongly depends on feature order!  The latest is retained

/**
 * 
 * Excludes features that have large (more than rMax) correlation coefficient
 * with other features. Order of features affects the result. For example
 * consider the following case: feature A has correlation with B more than rMax,
 * and B has correlation with C more than rMax, but at the same time, A has
 * correlation with C less than rMax. In this case if they have order A, B, C, A
 * will be excluded (as high-correlated with B) and B will be excluded (as
 * high-correlated with C). C will be retained. If they are ordered in B, A, C,
 * B will be excluded and A, C will be retained. The algorithm consequently
 * calculates correlation coefficient between each feature and each of
 * subsequent features and if it is more than rMax for one of subsequent
 * features, current feature is removed.
 *
 */
public class DropHighCorrPreprocessor extends DropFeaturesPreprocessor {

	private float rMax = 0.0F;

	@Override
	public void train(FeaturesGenerator features, ChemDataset data) {
		features.precompute(data);
		float[][] featuresFloat = features.features(data);
		ArrayList<Integer> drop = new ArrayList<Integer>();
		ArrayList<String> namesRetain = new ArrayList<String>();
		int size = featuresFloat.length;
		if (size * features.getNumFeatures() != 0) {
			float[][] featuresTransposed = ArUtls.transpose(featuresFloat);
			int n = features.getNumFeatures();
			float[][] corr = new float[n][n];
			Arrays.stream(ArUtls.intsrnd(n * n)).parallel().forEach(num -> {
				int i = num / n;
				int j = num % n;
				if (j > i) {
					float summI = 0;
					float summJ = 0;
					for (int k = 0; k < size; k++) {
						summI += featuresTransposed[i][k];
						summJ += featuresTransposed[j][k];
					}
					float averageI = summI / size;
					float averageJ = summJ / size;
					float summ1 = 0;
					float summ2 = 0;
					float summ3 = 0;

					for (int k = 0; k < size; k++) {
						summ1 += (featuresTransposed[i][k] - averageI) * (featuresTransposed[j][k] - averageJ);
						summ2 += (featuresTransposed[i][k] - averageI) * (featuresTransposed[i][k] - averageI);
						summ3 += (featuresTransposed[j][k] - averageJ) * (featuresTransposed[j][k] - averageJ);
					}
					float r = Float.NaN;
					if (summ2 * summ3 != 0) {
						r = summ1 / ((float) Math.sqrt(summ2 * summ3));
					} else {
						r = ((summ2 == 0) && (summ3 == 0)) ? 1.0f : 0.0f;
					}
					corr[i][j] = r;
				}
			});
			for (int i = 0; i < n; i++) {
				boolean dropF = false;
				for (int j = i + 1; j < n; j++) {
					if (Math.abs(corr[i][j]) >= rMax) {
						dropF = true;
					}
				}
				if (dropF) {
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
	 * 
	 * @param rMax maximum allowed correlation coefficient
	 */
	public DropHighCorrPreprocessor(float rMax) {
		this.rMax = rMax;
	}
}
