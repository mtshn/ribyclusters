package ru.ac.phyche.ribyclusters.models;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import ru.ac.phyche.ribyclusters.ArUtls;
import ru.ac.phyche.ribyclusters.ChemDataset;
import ru.ac.phyche.ribyclusters.ColumnFeatures;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;

public class LibSVMRI extends QSRRModelRI {

	private static final float epsTolerance = 1E-3F;
	private static final int nThreadsHyperparamsTune = 128;
	private static final float memCacheSizeMB = 256 * 1024;
	private static final QSRRModelRI.AccuracyMeasure accuracyMeasureTuning = QSRRModelRI.AccuracyMeasure.MDAE;
	private static final float minC = 1E-6f;
	private static final float maxC = 1.5E4f;
	private static final float minNu = 1E-3f;
	private static final float maxNu = 0.999f;
	private static final float minGamma = 7E-5f;
	private static final float maxGamma = 0.2f;
	private static final int maxNumberEntriesSVR = 12000;

	private svm_model mdl = null;
	private float c = 1.0f;
	private float nu = 0.5f;
	private float gamma = 0.003f;
	private boolean shrinking = false;

	public LibSVMRI(ColumnFeatures genCol, FeaturesGenerator gen) {
		super(genCol, gen);
	}

	public LibSVMRI(ColumnFeatures genCol, FeaturesGenerator gen, boolean hyperparamsTune, String hyperparamsTuneFile,
			int hyperparamsTuneAttempts) {
		super(genCol, gen, hyperparamsTune, hyperparamsTuneFile, hyperparamsTuneAttempts);
	}

	public LibSVMRI(ColumnFeatures genCol, FeaturesGenerator gen, float c, float nu) {
		super(genCol, gen);
		this.c = c;
		this.nu = nu;
		this.gamma = 1.0f / ((float) gen.getNumFeatures());
	}

	public LibSVMRI(ColumnFeatures genCol, FeaturesGenerator gen, float c, float nu, float gamma, boolean shrinking) {
		super(genCol, gen);
		this.c = c;
		this.nu = nu;
		this.gamma = gamma;
		this.shrinking = shrinking;
	}

	@Override
	public float[] predict(String[] smiles, int[] columns) {
		svm_node[][] f = datasetToFeaturesLibSVM(smiles, columns);
		float[] result = new float[smiles.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = (float) svm.svm_predict(mdl, f[i]) * 1000;
		}
		return result;
	}

	private svm_problem datasetToProblem(ChemDataset set) {
		float[][] features1 = features(set);
		svm_problem pr = ArUtls.toLibSVMFormat(features1, ArUtls.mult(0.001f, set.allRetentions()));
		return pr;
	}

	private svm_node[][] datasetToFeaturesLibLinear(ChemDataset set) {
		float[][] features1 = features(set);
		svm_node[][] f = ArUtls.toLibSVMFormat(features1);
		return f;
	}

	private float[][] features(ChemDataset set) {
		return features(set.allSmiles(), set.allColumns());
	}

	private float[][] features(String[] smiles, int[] columns) {
		getGen().precompute(smiles);
		float[][] features = getGen().features(smiles);
		float[][] features1 = new float[features.length][];
		float[][] columnsFeatures = getGenCol().columnFeatures(columns);
		for (int i = 0; i < features.length; i++) {
			features1[i] = ArUtls.mergeArrays(columnsFeatures[i], features[i]);
		}
		return features1;
	}

	private svm_node[][] datasetToFeaturesLibSVM(String[] smiles, int[] columns) {
		float[][] features1 = features(smiles, columns);
		svm_node[][] f = ArUtls.toLibSVMFormat(features1);
		return f;
	}

	private static class Param {
		float c;
		float nu;
		float gamma;
		boolean shrinking;

		private static float randomGamma(int nFeatures) {
			if (Math.random() > 0.5) {
				return 1.0f / ((float) nFeatures);
			} else {
				return (float) Math.pow(10,
						Math.log10(minGamma) + Math.random() * (Math.log10(maxGamma) - Math.log10(minGamma)));
			}
		}

		public static Param rnd(int nFeatures) {
			Param result = new Param();
			result.c = (float) Math.pow(10, Math.log10(minC) + Math.random() * (Math.log10(maxC) - Math.log10(minC)));
			result.gamma = randomGamma(nFeatures);
			result.nu = (float) (minNu + Math.random() * (maxNu - minNu));
			result.shrinking = Math.random() > 0.5;
			return result;
		}

		@Override
		public String toString() {
			return "C: " + c + " nu: " + nu + " gamma: " + gamma + " shrinking: " + shrinking;
		}
	}

	private svm_parameter prms(Param p) {
		svm_parameter params = new svm_parameter();
		params.nu = p.nu;
		params.svm_type = svm_parameter.NU_SVR;
		params.C = p.c;
		params.gamma = p.gamma;
		params.kernel_type = svm_parameter.RBF;
		params.cache_size = memCacheSizeMB;
		params.shrinking = p.shrinking ? 1 : 0;
		params.probability = 0;
		params.eps = epsTolerance;
		return params;
	}

	@Override
	public void train(ChemDataset trainSet, ChemDataset validationSet) {
		ChemDataset train = trainSet;
		if (trainSet.size() > maxNumberEntriesSVR) {
			ChemDataset train1 = train.copy().shuffle();
			train = train1.simpleShuffleSplit(maxNumberEntriesSVR);
		}
		svm_problem pr = datasetToProblem(train);
		if (isHyperparamsTune()) {
			try {
				int n = 0;
				svm_node[][] featuresVal = datasetToFeaturesLibLinear(validationSet);
				float bestAccuracy = Float.MAX_VALUE;
				Param bestParams = null;

				float[] labelsCorrect = validationSet.allRetentions();
				FileWriter fw = null;
				if (getHyperparamsTuneFile() != null) {
					fw = new FileWriter(getHyperparamsTuneFile());
				}
				while (n < getHyperparamsTuneAttempts()) {
					int x = Math.min(getHyperparamsTuneAttempts() - n, nThreadsHyperparamsTune);
					String[] accuracyMeasures = new String[x];
					Param[] params = new Param[x];
					Arrays.stream(ArUtls.intsrnd(x)).parallel().forEach(i -> {
						Param p = Param.rnd(getGen().getNumFeatures());
						svm_parameter parameter = prms(p);
						svm_model mdl1 = svm.svm_train(pr, parameter);
						float[] predictions = new float[featuresVal.length];
						for (int k = 0; k < predictions.length; k++) {
							predictions[k] = (float) svm.svm_predict(mdl1, featuresVal[k]) * 1000;
						}
						accuracyMeasures[i] = QSRRModelRI.accuracyMeasuresValidation(predictions, labelsCorrect);
						System.out.println(accuracyMeasures[i]);
						params[i] = p;
					});
					for (int i = 0; i < x; i++) {
						float accuracy = QSRRModelRI.accuracy(accuracyMeasureTuning, accuracyMeasures[i]);
						if (accuracy < bestAccuracy) {
							bestAccuracy = accuracy;
							bestParams = params[i];
						}
						if (fw != null) {
							fw.write(modelType() + " " + params[i].toString() + " " + accuracyMeasures[i] + "\n");
						}
					}
					if (fw != null) {
						fw.flush();
					}
					n = n + x;
				}
				if (fw != null) {
					fw.close();
				}
				c = bestParams.c;
				nu = bestParams.nu;
				gamma = bestParams.gamma;
				shrinking = bestParams.shrinking;
			} catch (IOException e) {
				e.printStackTrace();
				throw new RuntimeException(e.getMessage());
			}
		}
		Param p = new Param();
		p.c = c;
		p.gamma = gamma;
		p.nu = nu;
		p.shrinking = shrinking;
		svm_parameter params = prms(p);
		mdl = svm.svm_train(pr, params);
	}

	@Override
	public void save(String directory) throws IOException {
		Files.createDirectories(Paths.get(directory));
		FileWriter fw = new FileWriter(new File(directory, "ModelType.txt"));
		fw.write(this.modelType());
		fw.close();
		fw = new FileWriter(new File(directory, "info.txt"));
		fw.write(c + " " + nu + " " + gamma + " " + (shrinking ? 1 : 0));
		fw.close();
		fw = new FileWriter(new File(directory, "info1.txt"));
		fw.write(this.fullModelInfo());
		fw.close();
		File f = new File(directory, "model.txt");
		svm.svm_save_model(f.getAbsolutePath(), mdl);
	}

	@Override
	public void load(String directory) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(directory, "ModelType.txt")));
		String s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		if (!s.trim().equals(this.modelType())) {
			throw new RuntimeException("Wrong model type");
		}
		br = new BufferedReader(new FileReader(new File(directory, "info.txt")));
		s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		String[] split = s.trim().split("\\s+");
		this.c = Float.parseFloat(split[0]);
		this.nu = Float.parseFloat(split[1]);
		this.gamma = Float.parseFloat(split[2]);
		this.shrinking = Integer.parseInt(split[3]) > 0.5f;
		File f = new File(directory, "model.txt");
		mdl = svm.svm_load_model(f.getAbsolutePath());
	}

	@Override
	public String modelType() {
		return "LIBSVM";
	}

	@Override
	public String fullModelInfo() {
		return "LIBSVM " + " C: " + c + " nu: " + nu + " gamma: " + gamma + " shrinking: " + shrinking;
	}

	@Override
	public ModelRI createSimilar() {
		LibSVMRI result = (LibSVMRI) QSRRModelRI.getModel(this.modelType(), this.getGenCol(), this.getGen(),
				this.isHyperparamsTune(), this.getHyperparamsTuneFile(), this.getHyperparamsTuneAttempts());
		result.c = this.c;
		result.nu = this.nu;
		result.gamma = this.gamma;
		result.shrinking = this.shrinking;
		return result;
	}
}