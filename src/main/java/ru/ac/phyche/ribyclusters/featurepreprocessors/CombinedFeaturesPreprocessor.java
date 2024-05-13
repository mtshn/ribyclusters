package ru.ac.phyche.ribyclusters.featurepreprocessors;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import ru.ac.phyche.ribyclusters.ChemDataset;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;
import ru.ac.phyche.ribyclusters.featuregenerators.PreprocessedFeaturesGenerator;

/**
 * 
 * This preprocessor combines other preprocessors. The preprocessors are applied
 * in the same order as they were added using the addPreprocessor method. All
 * underlying preprocessor are trained (using the same data set and feature
 * generator) when train method is invoked. No explicit training of them is
 * required.
 *
 */
public class CombinedFeaturesPreprocessor extends FeaturesPreprocessor {

	private String[] names = new String[] {};
	private ArrayList<FeaturesPreprocessor> preprocessors = new ArrayList<FeaturesPreprocessor>();

	/**
	 * 
	 * @param n number of one of preprocessors
	 * @return preprocessor at number n
	 */
	public FeaturesPreprocessor getPreprocessor(int n) {
		return preprocessors.get(n);
	}

	/**
	 * 
	 * @return list of preprocessors
	 */
	public ArrayList<FeaturesPreprocessor> getPreprocessors() {
		return preprocessors;
	}

	/**
	 * 
	 * @return array of preprocessors
	 */
	public FeaturesPreprocessor[] getPreprocessorsArray() {
		return preprocessors.toArray(new FeaturesPreprocessor[preprocessors.size()]);
	}

	/**
	 * The preprocessors are applied in the same order as they were added
	 * 
	 * @param a a preprocessor (can be non-trained)
	 * @return this
	 */
	public CombinedFeaturesPreprocessor addPreprocessor(FeaturesPreprocessor a) {
		preprocessors.add(a);
		return this;
	}

	@Override
	public void train(FeaturesGenerator features, ChemDataset data) {
		CombinedFeaturesPreprocessor preproc = new CombinedFeaturesPreprocessor();
		preproc.names = features.getNames();
		for (int i = 0; i < preprocessors.size(); i++) {
			PreprocessedFeaturesGenerator x = new PreprocessedFeaturesGenerator(features, preproc);
			x.precompute(data);
			preprocessors.get(i).train(x, data);
			preproc.addPreprocessor(preprocessors.get(i));
			preproc.names = preprocessors.get(i).featureNames();
		}
		this.names = preprocessors.get(preprocessors.size() - 1).featureNames();
	}

	@Override
	public void save(FileWriter filewriter) throws IOException {
		filewriter.write("PREPROCESSOR CombinedFeaturesPreprocessor\n");
		filewriter.write("PREPROCESSOR CombinedFeaturesPreprocessor\n");
		for (int i = 0; i < preprocessors.size(); i++) {
			filewriter.write("PREPROCESSOR\n");
			preprocessors.get(i).save(filewriter);
			;
		}
		filewriter.write("END\n");
	}

	@Override
	public void load(BufferedReader filereader) throws IOException {
		preprocessors = new ArrayList<FeaturesPreprocessor>();
		String s = filereader.readLine();
		while (s.trim().equals("")) {
			s = filereader.readLine();
		}
		if (!s.trim().split("\\s+")[0].equals("PREPROCESSOR")) {
			throw (new IOException("Wrong file format! Word PREPROCESSOR is expected"));
		}
		if (!s.trim().split("\\s+")[1].equals("CombinedFeaturesPreprocessor")) {
			throw (new IOException("Wrong file format!"));
		}
		while (s.trim().split("\\s+")[0].equals("PREPROCESSOR") && s.trim().split("\\s+").length != 1
				&& s.trim().split("\\s+")[1].equals("CombinedFeaturesPreprocessor")) {
			s = filereader.readLine();
			while (s.trim().equals("")) {
				s = filereader.readLine();
			}
		}
		boolean read = true;
		while ((s != null) && read) {
			if (!s.trim().equals("")) {
				if (s.trim().equals("PREPROCESSOR")) {
					this.addPreprocessor(FeaturesPreprocessor.fromFile(filereader));
				} else {
					if (s.trim().equals("END")) {
						read = false;
					} else {
						throw (new IOException("Wrong file format!"));
					}
				}
			}
			if (read) {
				s = filereader.readLine();
			}
		}
		this.names = preprocessors.get(preprocessors.size() - 1).featureNames();
	}

	@Override
	public float[] preprocess(float[] input) {
		float[] rslt = input;
		for (int i = 0; i < preprocessors.size(); i++) {
			rslt = preprocessors.get(i).preprocess(rslt);
		}
		return rslt;
	}

	@Override
	public String[] featureNames() {
		return names;
	}

}
