package ru.ac.phyche.ribyclusters.models;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import ru.ac.phyche.ribyclusters.ColumnFeatures;
import ru.ac.phyche.ribyclusters.RIByClusters;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;

public abstract class QSRRModelRI extends ModelRI {
	public static enum AccuracyMeasure {
		MAE, MDAE, RMSE, MDPE, MPE
	};

	private FeaturesGenerator gen = null;
	private ColumnFeatures genCol = null;
	private boolean hyperparamsTune = false;
	private String hyperparamsTuneFile = null;
	private int hyperparamsTuneAttempts = 0;

	@Override
	public void setTuningOutFileOrDir(String filename) {
		this.setHyperparamsTuneFile(filename);
	}

	public FeaturesGenerator getGen() {
		return gen;
	}

	public void setGen(FeaturesGenerator gen) {
		this.gen = gen;
	}

	public ColumnFeatures getGenCol() {
		return genCol;
	}

	public void setGenCol(ColumnFeatures genCol) {
		this.genCol = genCol;
	}

	public boolean isHyperparamsTune() {
		return hyperparamsTune;
	}

	public void setHyperparamsTune(boolean hyperparamsTune) {
		this.hyperparamsTune = hyperparamsTune;
	}

	public String getHyperparamsTuneFile() {
		return hyperparamsTuneFile;
	}

	public void setHyperparamsTuneFile(String hyperparamsTuneFile) {
		this.hyperparamsTuneFile = hyperparamsTuneFile;
	}

	public int getHyperparamsTuneAttempts() {
		return hyperparamsTuneAttempts;
	}

	public void setHyperparamsTuneAttempts(int hyperparamsTuneAttempts) {
		this.hyperparamsTuneAttempts = hyperparamsTuneAttempts;
	}

	public QSRRModelRI(ColumnFeatures genCol, FeaturesGenerator gen) {
		this.gen = gen;
		this.genCol = genCol;
	}

	public QSRRModelRI(ColumnFeatures genCol, FeaturesGenerator gen, boolean hyperparamsTune,
			String hyperparamsTuneFile, int hyperparamsTuneAttempts) {
		this.gen = gen;
		this.genCol = genCol;
		this.hyperparamsTune = hyperparamsTune;
		this.hyperparamsTuneFile = hyperparamsTuneFile;
		this.hyperparamsTuneAttempts = hyperparamsTuneAttempts;
	}

	private static QSRRModelRI getModel(String modelType) {
		if (modelType.equals((new LibLinearRI(null, null)).modelType())) {
			return new LibLinearRI(null, null);
		}
		if (modelType.equals((new LibSVMRI(null, null)).modelType())) {
			return new LibSVMRI(null, null);
		}
		if (modelType.equals((new RidgeRI(null, null)).modelType())) {
			return new RidgeRI(null, null);
		}
		if (modelType.equals((new XgbRI(null, null)).modelType())) {
			return new XgbRI(null, null);
		}
		throw new RuntimeException("Unknown model type " + modelType);
	}

	public static QSRRModelRI getModel(String modelType, ColumnFeatures genCol, FeaturesGenerator gen) {
		QSRRModelRI result = getModel(modelType);
		result.gen = gen;
		result.genCol = genCol;
		return result;
	}

	public static QSRRModelRI getModel(String modelType, ColumnFeatures genCol, FeaturesGenerator gen,
			boolean hyperparamsTune, String hyperparamsTuneFile, int hyperparamsTuneAttempts) {
		QSRRModelRI result = getModel(modelType);
		result.gen = gen;
		result.genCol = genCol;
		result.hyperparamsTune = hyperparamsTune;
		result.hyperparamsTuneFile = hyperparamsTuneFile;
		result.hyperparamsTuneAttempts = hyperparamsTuneAttempts;
		return result;
	}

	public static QSRRModelRI loadModel(String directoryName, ColumnFeatures genCol, FeaturesGenerator gen)
			throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(directoryName, "ModelType.txt")));
		String s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		QSRRModelRI result = getModel(s.trim(), genCol, gen);
		result.load(directoryName);
		return result;
	}
}
