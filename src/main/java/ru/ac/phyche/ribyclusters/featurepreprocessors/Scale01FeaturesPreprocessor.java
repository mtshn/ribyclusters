package ru.ac.phyche.ribyclusters.featurepreprocessors;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;

import ru.ac.phyche.ribyclusters.ChemDataset;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;

/**
 * 
 * Scale all features to 0...1 range. Each value is converted to
 * ((val-min)/(max-min)) where min - is the minimum value of the given feature
 * in the training set and max - is the maximum value of the given feature in
 * the training set. If min==max result is zero (not NaN). File format. The
 * first line: "PREPROCESSOR Scale01FeaturesPreprocessor". The second line is
 * the same as the first one: "PREPROCESSOR Scale01FeaturesPreprocessor". The
 * third one contains one integer number: number of features. The fourth line
 * contains space-separated names of the features). The 5th line contains (space
 * separated) min values for all features. The 6th line contains (space
 * separated) max values for all features. The 7th line: "END".
 *
 */
public class Scale01FeaturesPreprocessor extends FeaturesPreprocessor {

	private float[] min = new float[] {};
	private float[] max = new float[] {};
	private String[] names = null;

	/**
	 * 
	 * @return array of min values for all features
	 */
	public float[] getMin() {
		return min;
	}

	/**
	 * 
	 * @return array of max values for all features
	 */
	public float[] getMax() {
		return max;
	}

	/**
	 * 
	 * @param val value
	 * @param min min value for the feature
	 * @param max max value for the feature
	 * @return ((val - min) / (max - min)) and 0 if min == max
	 */
	public static float scaleTo01(float val, float min, float max) {
		if (max - min == 0) {
			return 0f;
		}
		return ((val - min) / (max - min));
	}

	@Override
	public void train(FeaturesGenerator features, ChemDataset data) {
		features.precompute(data);
		float[][] featuresFloat = features.features(data);
		this.min = new float[features.getNumFeatures()];
		this.max = new float[features.getNumFeatures()];
		if (data.size() == 0) {
			throw (new RuntimeException("Train set has zero size"));
		}
		if (featuresFloat.length * features.getNumFeatures() != 0) {
			for (int i = 0; i < features.getNumFeatures(); i++) {
				float min_ = Float.POSITIVE_INFINITY;
				float max_ = Float.NEGATIVE_INFINITY;
				for (int j = 0; j < featuresFloat.length; j++) {
					if (featuresFloat[j][i] < min_) {
						min_ = featuresFloat[j][i];
					}
					if (featuresFloat[j][i] > max_) {
						max_ = featuresFloat[j][i];
					}
				}
				this.min[i] = min_;
				this.max[i] = max_;
			}
		}
		names = features.getNames();
	}

	@Override
	public void save(FileWriter filewriter) throws IOException {
		filewriter.write("PREPROCESSOR Scale01FeaturesPreprocessor\n");
		filewriter.write("PREPROCESSOR Scale01FeaturesPreprocessor\n");
		filewriter.write(names.length + "\n");
		for (int i = 0; i < names.length; i++) {
			filewriter.write(names[i] + " ");
		}
		filewriter.write("\n");
		for (int i = 0; i < min.length; i++) {
			filewriter.write(min[i] + " ");
		}
		filewriter.write("\n");
		for (int i = 0; i < max.length; i++) {
			filewriter.write(max[i] + " ");
		}
		filewriter.write("\n");
		filewriter.write("END\n");
	}

	@Override
	public void load(BufferedReader filereader) throws IOException {
		String s = filereader.readLine();
		while (s.trim().equals("")) {
			s = filereader.readLine();
		}
		if (!s.trim().split("\\s+")[0].equals("PREPROCESSOR")) {
			throw (new IOException("Wrong file format! Word PREPROCESSOR is expected"));
		}
		if (!s.trim().split("\\s+")[1].equals("Scale01FeaturesPreprocessor")) {
			throw (new IOException("Wrong file format!"));
		}
		while (s.trim().split("\\s+")[0].equals("PREPROCESSOR")
				&& s.trim().split("\\s+")[1].equals("Scale01FeaturesPreprocessor")) {
			s = filereader.readLine();
			while (s.trim().equals("")) {
				s = filereader.readLine();
			}
		}
		if (s.trim().equals("0")) {
			names = new String[] {};
			min = new float[] {};
			max = new float[] {};
		} else {
			int n = Integer.parseInt(s.trim());
			s = filereader.readLine();
			while (s.trim().equals("")) {
				s = filereader.readLine();
			}
			names = s.trim().split("\\s+");
			if (names.length != n) {
				throw (new IOException("Wrong file format!"));
			}

			s = filereader.readLine();
			while (s.trim().equals("")) {
				s = filereader.readLine();
			}
			String[] floats = s.trim().split("\\s+");
			min = new float[floats.length];
			for (int i = 0; i < min.length; i++) {
				min[i] = Float.parseFloat(floats[i]);
			}
			s = filereader.readLine();
			while (s.trim().equals("")) {
				s = filereader.readLine();
			}
			floats = s.trim().split("\\s+");
			max = new float[floats.length];
			for (int i = 0; i < max.length; i++) {
				max[i] = Float.parseFloat(floats[i]);
			}
		}
		s = filereader.readLine();
		while (s.trim().equals("")) {
			s = filereader.readLine();
		}
		if (!s.trim().equals("END")) {
			throw (new IOException("Wrong file format!"));
		}
		if ((min.length != names.length) || (max.length != names.length)) {
			throw (new IOException("Wrong file format!"));
		}
	}

	@Override
	public float[] preprocess(float[] input) {
		if (input.length != names.length) {
			throw (new RuntimeException("Wrong length of array! Preprocessing failed"));
		}
		float[] outp = new float[input.length];
		for (int i = 0; i < outp.length; i++) {
			outp[i] = scaleTo01(input[i], min[i], max[i]);
		}
		return outp;
	}

	@Override
	public String[] featureNames() {
		return names;
	}

}
