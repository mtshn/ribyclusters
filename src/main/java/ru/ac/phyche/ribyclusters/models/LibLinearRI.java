package ru.ac.phyche.ribyclusters.models;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import ru.ac.phyche.ribyclusters.ArUtls;
import ru.ac.phyche.ribyclusters.ChemDataset;
import ru.ac.phyche.ribyclusters.ColumnFeatures;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;

public class LibLinearRI extends QSRRModelRI {

	private static final float epsTolerance = 5E-3F;
	private static final int maxIterations = 3000;
	private static final int nThreadsHyperparamsTune = 128;
	private static final QSRRModelRI.AccuracyMeasure accuracyMeasureTuning = QSRRModelRI.AccuracyMeasure.MDAE;
	private static final float minC = 1E-6f;
	private static final float maxC = 1E5f;
	private static final float minp = 1E-7f;
	private static final float maxp = 2f;

	private Model mdl = null;
	private float c = 1.0f;
	private float p = 0.1f;
	private int solverType = 2;

	public LibLinearRI(ColumnFeatures genCol, FeaturesGenerator gen) {
		super(genCol, gen);
	}

	public LibLinearRI(ColumnFeatures genCol, FeaturesGenerator gen, boolean hyperparamsTune,
			String hyperparamsTuneFile, int hyperparamsTuneAttempts) {
		super(genCol, gen, hyperparamsTune, hyperparamsTuneFile, hyperparamsTuneAttempts);
	}

	public LibLinearRI(ColumnFeatures genCol, FeaturesGenerator gen, float c, float p) {
		super(genCol, gen);
		this.c = c;
		this.p = p;
	}

	public LibLinearRI(ColumnFeatures genCol, FeaturesGenerator gen, float c, float p, int solverType) {
		super(genCol, gen);
		this.c = c;
		this.p = p;
		if ((solverType <= 0) || (solverType > 2)) {
			throw new RuntimeException("Correct values for solverType are 1,2,3");
		}
		this.solverType = solverType;
	}

	private SolverType st() {
		return (solverType == 0) ? SolverType.L2R_L1LOSS_SVR_DUAL
				: ((solverType == 1) ? SolverType.L2R_L2LOSS_SVR : SolverType.L2R_L2LOSS_SVR_DUAL);
	}

	private static SolverType st(int solverType) {
		return (solverType == 0) ? SolverType.L2R_L1LOSS_SVR_DUAL
				: ((solverType == 1) ? SolverType.L2R_L2LOSS_SVR : SolverType.L2R_L2LOSS_SVR_DUAL);
	}

	@Override
	public float[] predict(String[] smiles, int[] columns) {
		Feature[][] f = datasetToFeaturesLibLinear(smiles, columns);
		float[] result = new float[smiles.length];
		for (int i = 0; i < result.length; i++) {
			result[i] = (float) Linear.predict(mdl, f[i]) * 1000;
		}
		return result;
	}

	private Problem datasetToProblem(ChemDataset set) {
		float[][] features1 = features(set);
		Problem pr = ArUtls.toLibLinearFormat(features1, ArUtls.mult(0.001f, set.allRetentions()));
		return pr;
	}

	private Feature[][] datasetToFeaturesLibLinear(ChemDataset set) {
		float[][] features1 = features(set);
		Feature[][] f = ArUtls.toLibLinearFormat(features1);
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

	private Feature[][] datasetToFeaturesLibLinear(String[] smiles, int[] columns) {
		float[][] features1 = features(smiles, columns);
		Feature[][] f = ArUtls.toLibLinearFormat(features1);
		return f;
	}

	private static class Param {
		float c;
		float p;
		int st;

		public static Param rnd() {
			Param result = new Param();
			result.c = (float) Math.pow(10, Math.log10(minC) + Math.random() * (Math.log10(maxC) - Math.log10(minC)));
			result.p = (float) Math.pow(10, Math.log10(minp) + Math.random() * (Math.log10(maxp) - Math.log10(minp)));
			result.st = (int) Math.round(Math.random() * 3 - 0.5);
			return result;
		}

		@Override
		public String toString() {
			return "C: " + c + " p: " + p + " solver_type: " + st;
		}
	}

	@Override
	public void train(ChemDataset trainSet, ChemDataset validationSet) {
		Problem pr = datasetToProblem(trainSet);
		if (this.isHyperparamsTune()) {
			try {
				int n = 0;
				Feature[][] featuresVal = datasetToFeaturesLibLinear(validationSet);
				float bestAccuracy = Float.MAX_VALUE;
				Param bestParams = null;
				String[] smiles = validationSet.allSmiles();

				float[] labelsCorrect = validationSet.allRetentions();
				FileWriter fw = null;
				if (getHyperparamsTuneFile() != null) {
					fw = new FileWriter(getHyperparamsTuneFile());
				}
				while (n < getHyperparamsTuneAttempts()) {
					int x = Math.min(getHyperparamsTuneAttempts() - n, nThreadsHyperparamsTune);
					float[][] predictionsVal = new float[x][];
					Param[] params = new Param[x];
					Arrays.stream(ArUtls.intsrnd(x)).parallel().forEach(i -> {
						Param p = Param.rnd();
						Parameter parameter = new Parameter(st(p.st), p.c, epsTolerance, maxIterations, p.p);
						Model mdl1 = Linear.train(pr, parameter);
						float[] predictions = new float[featuresVal.length];
						for (int k = 0; k < predictions.length; k++) {
							predictions[k] = (float) Linear.predict(mdl1, featuresVal[k]) * 1000;
						}
						predictionsVal[i] = predictions;
						params[i] = p;
					});
					for (int i = 0; i < x; i++) {
						String accuracyMeasures = QSRRModelRI.accuracyMeasuresValidation(smiles, predictionsVal[i],
								labelsCorrect, false, null);
						float accuracy = QSRRModelRI.accuracy(accuracyMeasureTuning, accuracyMeasures);
						if (accuracy < bestAccuracy) {
							bestAccuracy = accuracy;
							bestParams = params[i];
						}
						if (fw != null) {
							fw.write(modelType() + " " + params[i].toString() + " " + accuracyMeasures + "\n");
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
				p = bestParams.p;
				solverType = bestParams.st;
			} catch (IOException e) {
				e.printStackTrace();
				throw new RuntimeException(e.getMessage());
			}
		}
		Parameter params = new Parameter(st(), c, epsTolerance, maxIterations, p);
		mdl = Linear.train(pr, params);
	}

	@Override
	public void save(String directory) throws IOException {
		Files.createDirectories(Paths.get(directory));
		FileWriter fw = new FileWriter(new File(directory, "ModelType.txt"));
		fw.write(this.modelType());
		fw.close();
		fw = new FileWriter(new File(directory, "info.txt"));
		fw.write(c + " " + p + " " + solverType);
		fw.close();
		fw = new FileWriter(new File(directory, "info1.txt"));
		fw.write(this.fullModelInfo());
		fw.close();
		File f = new File(directory, "model.txt");
		mdl.save(f.toPath());
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
		this.p = Float.parseFloat(split[1]);
		this.solverType = Integer.parseInt(split[2]);
		File f = new File(directory, "model.txt");
		mdl = Model.load(f.toPath());
	}

	@Override
	public String modelType() {
		return "LIBLINEAR";
	}

	@Override
	public String fullModelInfo() {
		return "LIBLINEAR " + " C: " + c + " p: " + p + " solver_type: " + solverType;
	}

	@Override
	public ModelRI createSimilar() {
		LibLinearRI result = (LibLinearRI) QSRRModelRI.getModel(this.modelType(), this.getGenCol(), this.getGen(),
				this.isHyperparamsTune(), this.getHyperparamsTuneFile(), this.getHyperparamsTuneAttempts());
		result.c = this.c;
		result.p = this.p;
		result.solverType = this.solverType;
		return result;
	}

}