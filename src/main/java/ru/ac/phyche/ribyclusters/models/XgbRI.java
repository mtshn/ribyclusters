package ru.ac.phyche.ribyclusters.models;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;

import org.apache.commons.lang3.tuple.Pair;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ru.ac.phyche.ribyclusters.ArUtls;
import ru.ac.phyche.ribyclusters.ChemDataset;
import ru.ac.phyche.ribyclusters.ColumnFeatures;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;

public class XgbRI extends QSRRModelRI {

	private static final QSRRModelRI.AccuracyMeasure accuracyMeasureTuning = QSRRModelRI.AccuracyMeasure.MDAE;
	private static final int useHistMethodWithMoreThanXSamples = 1000;
	private static final int validateEveryXTrees = 50;
	private static final int stopIfLossDoesntDecreaseXIters = 249;
	private static final float considerableRelativeLossDecrease = 0.01f;

	private static final int maxTrees = 5000;
	private static final int[] maxDepthRange = new int[] { 1, 24 };
	private static final int[] minChildWeightRange = new int[] { 1, 24 };
	private static final float[] etaRange = new float[] { 0.01f, 0.45f };
	private static final float[] lambdaRange = new float[] { 0.01f, 15f };
	private static final float[] gammaRange = new float[] { 0.01f, 5f };
	private static final float[] subsampleRange = new float[] { 0.3f, 1.0f };

	private Booster mdl = null;
	private float eta = 0.3f;
	private float gamma = 0.0f;
	private float lambda = 1.0f;
	private float subsample = 0.9f;
	private int maxDepth = 6;
	private int minChildWeight = 1;

	private int nTrees = -1;

	public XgbRI(ColumnFeatures genCol, FeaturesGenerator gen) {
		super(genCol, gen);
	}

	public XgbRI(ColumnFeatures genCol, FeaturesGenerator gen, boolean hyperparamsTune, String hyperparamsTuneFile,
			int hyperparamsTuneAttempts) {
		super(genCol, gen, hyperparamsTune, hyperparamsTuneFile, hyperparamsTuneAttempts);
	}

	public XgbRI(ColumnFeatures genCol, FeaturesGenerator gen, int minChildWeight, int maxDepth) {
		super(genCol, gen);
		this.minChildWeight = minChildWeight;
		this.maxDepth = maxDepth;
	}

	public XgbRI(ColumnFeatures genCol, FeaturesGenerator gen, int minChildWeight, int maxDepth, float eta, float gamma,
			float lambda, float subsample) {
		super(genCol, gen);
		this.minChildWeight = minChildWeight;
		this.maxDepth = maxDepth;
		this.eta = eta;
		this.gamma = gamma;
		this.lambda = lambda;
	}

	private static class Param {
		float eta = 0.3f;
		float gamma = 0.0f;
		float lambda = 1.0f;
		float subsample = 0.9f;
		int maxDepth = 6;
		int minChildWeight = 1;

		private static float logrnd(float[] minmax) {
			double log10min = Math.log10(minmax[0]);
			double log10max = Math.log10(minmax[1]);
			double r = log10min + Math.random() * (log10max - log10min);
			r = Math.pow(10, r);
			return (float) r;
		}

		private static float rnd(float[] minmax) {
			double r = minmax[0] + Math.random() * (minmax[1] - minmax[0]);
			return (float) r;
		}

		private static int rnd(int[] minmax) {
			double r = minmax[0] - 0.5 + Math.random() * (minmax[1] - minmax[0] + 1);
			return Math.round((float) r);
		}

		public static Param rnd() {
			Param result = new Param();
			result.eta = logrnd(etaRange);
			result.gamma = logrnd(gammaRange);
			result.lambda = logrnd(lambdaRange);
			result.subsample = rnd(subsampleRange);
			result.maxDepth = rnd(maxDepthRange);
			result.minChildWeight = rnd(minChildWeightRange);
			return result;
		}

		@Override
		public String toString() {
			String s = "";
			s = s + "eta: " + eta;
			s = s + " gamma: " + gamma;
			s = s + " lambda: " + lambda;
			s = s + " subsample: " + subsample;
			s = s + " maxDepth: " + maxDepth;
			s = s + " minChildWeight: " + minChildWeight;
			return s;
		}
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

	private DMatrix trainingSet(ChemDataset trainSet) throws IOException {
		float[][] features = features(trainSet);
		float[] labels = ArUtls.mult(0.001f, trainSet.allRetentions());
		return ArUtls.dataSetToXGBooostDMatrix(features, labels);
	}

	private DMatrix featuresdm(ChemDataset set) throws XGBoostError {
		float[][] features = features(set);
		DMatrix f = new DMatrix(ArUtls.flatten(features), features.length, features[0].length, -777f);
		return f;
	}

	private Pair<Booster, Integer> train(Param p, DMatrix trainSet, DMatrix testSet, float[] testSetLabels,
			int trainSetSize) throws XGBoostError, IOException {
		HashMap<String, Object> hm = new HashMap<String, Object>();
		hm.put("eta", p.eta);
		hm.put("gamma", p.gamma);
		hm.put("lambda", p.lambda);
		hm.put("max_depth", p.maxDepth);
		hm.put("min_child_weight", p.minChildWeight);
		hm.put("subsample", p.subsample);
		hm.put("objective", "reg:pseudohubererror");
		if (trainSetSize > useHistMethodWithMoreThanXSamples) {
			hm.put("tree_method", "hist");
		}
		hm.put("objective", "reg:pseudohubererror");
		Booster x = XGBoost.train(trainSet, hm, 1, new HashMap<String, DMatrix>(), null, null);
		boolean contin = true;
		float bestAccuracy = Float.MAX_VALUE;
		int bestAccuracyIter = 0;
		int iter = 1;
		while (contin) {
			for (int i = 0; i < validateEveryXTrees; i++) {
				x.update(trainSet, iter);
				iter++;
			}
			float accuracy = QSRRModelRI.accuracy(accuracyMeasureTuning, validate(x, testSet, testSetLabels, iter));
			if (accuracy < bestAccuracy * (1 - considerableRelativeLossDecrease)) {
				bestAccuracy = accuracy;
				bestAccuracyIter = iter;
			}
			if (iter - bestAccuracyIter >= stopIfLossDoesntDecreaseXIters) {
				contin = false;
			}
			if (iter >= XgbRI.maxTrees) {
				contin = false;
			}
		}
		return Pair.of(x, iter);
	}

	private String validate(Booster x, DMatrix testSet, float[] testSetLabels, int iterNum)
			throws IOException, XGBoostError {
		float[] predictions = ArUtls.mult(1000, ArUtls.flatten(x.predict(testSet)));
		String accuracyMeasures = QSRRModelRI.accuracyMeasuresValidation(predictions, testSetLabels);
		System.out.println("XGBOOST training. Iteration " + iterNum + " | " + accuracyMeasures);
		return accuracyMeasures;
	}

	@Override
	public void train(ChemDataset trainSet, ChemDataset validationSet) {
		try {
			DMatrix traindm = this.trainingSet(trainSet);
			DMatrix valdm = featuresdm(validationSet);
			float[] valRI = validationSet.allRetentions();
			if (isHyperparamsTune()) {
				float bestAccuracy = Float.MAX_VALUE;
				Param bestParams = null;
				FileWriter fw = null;
				if (getHyperparamsTuneFile() != null) {
					fw = new FileWriter(getHyperparamsTuneFile());
				}
				for (int n = 0; n < getHyperparamsTuneAttempts(); n++) {
					Param p = Param.rnd();
					Pair<Booster, Integer> b = this.train(p, traindm, valdm, valRI, trainSet.size());
					String accuracyMeasures = this.validate(b.getLeft(), valdm, valRI, b.getRight());
					float accuracy = QSRRModelRI.accuracy(accuracyMeasureTuning, accuracyMeasures);
					if (accuracy < bestAccuracy) {
						bestAccuracy = accuracy;
						bestParams = p;
					}
					if (fw != null) {
						fw.write(modelType() + " " + p.toString() + " " + accuracyMeasures + "\n");
						fw.flush();
					}
					b.getLeft().dispose();
				}

				if (fw != null) {
					fw.close();
				}
				this.eta = bestParams.eta;
				this.gamma = bestParams.gamma;
				this.lambda = bestParams.lambda;
				this.subsample = bestParams.subsample;
				this.maxDepth = bestParams.maxDepth;
				this.minChildWeight = bestParams.minChildWeight;
			}
			Param p = new Param();
			p.eta = this.eta;
			p.gamma = this.gamma;
			p.lambda = this.lambda;
			p.subsample = this.subsample;
			p.maxDepth = this.maxDepth;
			p.minChildWeight = this.minChildWeight;
			Pair<Booster, Integer> b = this.train(p, traindm, valdm, valRI, trainSet.size());
			this.mdl = b.getKey();
			this.nTrees = b.getRight();
			traindm.dispose();
			valdm.dispose();
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException(e.getMessage());
		} catch (XGBoostError e) {
			e.printStackTrace();
			throw new RuntimeException(e.getMessage());
		}
	}

	@Override
	public float[] predict(String[] smiles, int[] columns) {
		try {
			float[][] features = features(smiles, columns);
			DMatrix f = new DMatrix(ArUtls.flatten(features), features.length, features[0].length, -777f);
			float[] predictions = ArUtls.mult(1000, ArUtls.flatten(mdl.predict(f)));
			return predictions;
		} catch (XGBoostError e) {
			e.printStackTrace();
			throw new RuntimeException(e.getMessage());
		}
	}

	@Override
	public void save(String directory) throws IOException {
		Files.createDirectories(Paths.get(directory));
		FileWriter fw = new FileWriter(new File(directory, "ModelType.txt"));
		fw.write(this.modelType());
		fw.close();
		fw = new FileWriter(new File(directory, "info.txt"));
		fw.write(eta + " " + gamma + " " + lambda + " " + subsample + " " + maxDepth + " " + minChildWeight);
		fw.close();
		fw = new FileWriter(new File(directory, "info1.txt"));
		fw.write(this.fullModelInfo());
		fw.close();
		File f = new File(directory, "model");
		try {
			mdl.saveModel(f.getAbsolutePath() + ".json");
		} catch (XGBoostError e) {
			e.printStackTrace();
			throw new IOException(e.getMessage());
		}
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
		this.eta = Float.parseFloat(split[0]);
		this.gamma = Float.parseFloat(split[1]);
		this.lambda = Float.parseFloat(split[2]);
		this.subsample = Float.parseFloat(split[3]);

		this.maxDepth = Integer.parseInt(split[4]);
		this.minChildWeight = Integer.parseInt(split[5]);

		File f = new File(directory, "model");
		try {
			mdl = XGBoost.loadModel(f.getAbsolutePath() + ".json");
		} catch (XGBoostError e) {
			e.printStackTrace();
			throw new IOException(e.getMessage());
		}

	}

	@Override
	public String modelType() {
		return "XGBOOST";
	}

	@Override
	public String fullModelInfo() {
		String s = "";
		s = s + "eta: " + eta;
		s = s + " gamma: " + gamma;
		s = s + " lambda: " + lambda;
		s = s + " subsample: " + subsample;
		s = s + " maxDepth: " + maxDepth;
		s = s + " minChildWeight: " + minChildWeight;
		s = s + " ";
		return "XGBOOST " + s + " nTrees " + nTrees;
	}

	@Override
	public ModelRI createSimilar() {
		XgbRI result = (XgbRI) QSRRModelRI.getModel(this.modelType(), this.getGenCol(), this.getGen(),
				this.isHyperparamsTune(), this.getHyperparamsTuneFile(), this.getHyperparamsTuneAttempts());
		result.eta = this.eta;
		result.gamma = this.gamma;
		result.lambda = this.lambda;
		result.subsample = this.subsample;
		result.maxDepth = this.maxDepth;
		result.minChildWeight = this.minChildWeight;
		return result;
	}
}
