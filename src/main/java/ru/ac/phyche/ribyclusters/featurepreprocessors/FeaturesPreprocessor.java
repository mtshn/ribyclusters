package ru.ac.phyche.ribyclusters.featurepreprocessors;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;

import ru.ac.phyche.ribyclusters.ChemDataset;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;

/**
 * Subclasses of this class implement feature preprocessors. Features generated
 * using FeatureGenerators can be preprocessed. Almost often preprocessor should
 * be trained using a train set and than can be applied to features. Simple
 * example how it does works. Scale01FeaturesPreprocessor learns scale factor
 * that should be applied to scale values of each feature to the 0...1 range.
 * For each of features scaling factors are different. Than the trained
 * preprocessor can be applied to test data.
 *
 */
public abstract class FeaturesPreprocessor {
	/**
	 * Train the generator using the training set. Intended behavior: this methods
	 * precomputes features in the FeaturesGenerator of each compound from data.
	 * 
	 * @param features FeatureGenerator that provides non-preprocessed feature.
	 * @param data     training set. Both SMILES and values are used.
	 */
	public abstract void train(FeaturesGenerator features, ChemDataset data);

	/**
	 * Save trained using the training set preprocessor. Then it can be loaded and
	 * used for test data. The FileWriter must be opened and closed later. This
	 * method does not close or open the FileWriter.
	 * 
	 * @param filewriter filewriter (must be opened).
	 * @throws IOException io
	 */
	public abstract void save(FileWriter filewriter) throws IOException;

	/**
	 * Load trained preprocessor from the file. The BufferedReader must be opened
	 * and closed later. This method does not close or open the BufferedReader
	 * 
	 * @param filereader buffered reader (must be opened!)
	 * @throws IOException io
	 */
	public abstract void load(BufferedReader filereader) throws IOException;

	/**
	 * Apply the preprocessor to a features (for one compound)
	 * 
	 * @param input features
	 * @return preprocessed features
	 */
	public abstract float[] preprocess(float[] input);

	/**
	 * Access to names of features.
	 * 
	 * @return feature names
	 */
	public abstract String[] featureNames();

	/**
	 * Load trained preprocessor from the file. The BufferedReader must be opened
	 * and closed later. This method does not close or open the BufferedReader.
	 * Training is not possible after loading from file.
	 * 
	 * @param filereader filewriter filewriter (must be opened).
	 * @return loaded FeaturePreprocessor
	 * @throws IOException io
	 */
	public static FeaturesPreprocessor fromFile(BufferedReader filereader) throws IOException {
		String s = filereader.readLine();
		while (s.trim().equals("")) {
			s = filereader.readLine();
		}
		if (!s.trim().split("\\s+")[0].equals("PREPROCESSOR")) {
			throw (new IOException("Wrong file format! Word PREPROCESSOR is expected"));
		}
		if (s.trim().split("\\s+")[1].equals("DropFeaturesPreprocessor")) {
			FeaturesPreprocessor r = new DropFeaturesPreprocessorNoTrain();
			r.load(filereader);
			return r;
		}
		if (s.trim().split("\\s+")[1].equals("CombinedFeaturesPreprocessor")) {
			FeaturesPreprocessor r = new CombinedFeaturesPreprocessor();
			r.load(filereader);
			return r;
		}
		if (s.trim().split("\\s+")[1].equals("Scale01FeaturesPreprocessor")) {
			FeaturesPreprocessor r = new Scale01FeaturesPreprocessor();
			r.load(filereader);
			return r;
		}
		if (s.trim().split("\\s+")[1].equals("ReplaceNaNsPreprocessor")) {
			FeaturesPreprocessor r = new ReplaceNaNsPreprocessor();
			r.load(filereader);
			return r;
		}
		throw (new IOException("Wrong file format! Unknown preprocesson name"));
	}

}
