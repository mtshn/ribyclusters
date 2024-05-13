package ru.ac.phyche.ribyclusters.models;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import com.thoughtworks.xstream.XStream;
import com.thoughtworks.xstream.io.xml.StaxDriver;

import ru.ac.phyche.ribyclusters.ArUtls;
import ru.ac.phyche.ribyclusters.ChemDataset;
import ru.ac.phyche.ribyclusters.ColumnFeatures;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.regression.LinearModel;
import smile.regression.RidgeRegression;

public class RidgeRI extends QSRRModelRI {
	private static final QSRRModelRI.AccuracyMeasure accuracyMeasureTuning = QSRRModelRI.AccuracyMeasure.MDAE;

	private static final float[] l2Range = new float[] { 1E-9f, 1E5f };

	private LinearModel mdl = null;
	private float l2 = 0.0001f;

	public RidgeRI(ColumnFeatures genCol, FeaturesGenerator gen) {
		super(genCol, gen);
	}

	public RidgeRI(ColumnFeatures genCol, FeaturesGenerator gen, boolean hyperparamsTune, String hyperparamsTuneFile,
			int hyperparamsTuneAttempts) {
		super(genCol, gen, hyperparamsTune, hyperparamsTuneFile, hyperparamsTuneAttempts);
	}

	public RidgeRI(ColumnFeatures genCol, FeaturesGenerator gen, float l2) {
		super(genCol, gen);
		this.l2 = l2;
	}

	private static class Param {
		float l2 = 0.0001f;

		private static float logrnd(float[] minmax) {
			double log10min = Math.log10(minmax[0]);
			double log10max = Math.log10(minmax[1]);
			double r = log10min + Math.random() * (log10max - log10min);
			r = Math.pow(10, r);
			return (float) r;
		}

		public static Param rnd() {
			Param result = new Param();
			result.l2 = logrnd(l2Range);
			return result;
		}

		@Override
		public String toString() {
			String s = "";
			s = s + "l2: " + l2;
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

	private LinearModel train(Param p, float[][] trainFeatures, float[] labels) throws IOException {
		for (int j = 0; j < trainFeatures[0].length; j++) {// Adding noise to constant columns!!!!
			float x = trainFeatures[0][j];
			boolean isconstant = true;
			for (int i = 0; i < trainFeatures.length; i++) {
				if (Math.abs(trainFeatures[i][j] - x) > 1E-5F) {
					isconstant = false;
				}
			}
			if (isconstant) {
				int rnd = (int) Math.round(Math.random() * trainFeatures.length);
				trainFeatures[rnd][j] += 2E-5;
			}
		}

		DataFrame dataFrame = ArUtls.toDataFrame(trainFeatures, ArUtls.mult(0.001F, labels));
		try {
			return RidgeRegression.fit(Formula.lhs("label"), dataFrame, p.l2);
		} catch (Throwable e) {
			e.printStackTrace();
			return null;
		}
	}

	private String validate(LinearModel lm, float[][] features, float[] labels) {
		float[] predictions = ArUtls.mult(1000, ArUtls.toFloatArray(lm.predict(ArUtls.toDataFrame(features))));
		String accuracyMeasures = QSRRModelRI.accuracyMeasuresValidation(predictions, labels);
		System.out.println(accuracyMeasures);
		return accuracyMeasures;
	}

	@Override
	public void train(ChemDataset trainSet, ChemDataset validationSet) {
		try {
			float[][] trainFeatures = this.features(trainSet);
			float[][] validationFeatures = this.features(validationSet);
			float[] trainLabels = trainSet.allRetentions();
			float[] validationLabels = validationSet.allRetentions();
			if (isHyperparamsTune()) {
				float bestAccuracy = Float.MAX_VALUE;
				Param bestParams = null;

				FileWriter fw = null;
				if (getHyperparamsTuneFile() != null) {
					fw = new FileWriter(getHyperparamsTuneFile());
				}
				for (int n = 0; n < getHyperparamsTuneAttempts(); n++) {
					Param p = Param.rnd();
					LinearModel b = this.train(p, trainFeatures, trainLabels);
					if (b != null) {
						String accuracyMeasures = this.validate(b, validationFeatures, validationLabels);
						float accuracy = QSRRModelRI.accuracy(accuracyMeasureTuning, accuracyMeasures);
						if (accuracy < bestAccuracy) {
							bestAccuracy = accuracy;
							bestParams = p;
						}
						if (fw != null) {
							fw.write(modelType() + " " + p.toString() + " " + accuracyMeasures + "\n");
							fw.flush();
						}
					} else {
						fw.write(modelType() + " " + p.toString() + " Training Failed\n");
					}
				}

				if (fw != null) {
					fw.close();
				}
				this.l2 = bestParams.l2;
			}
			Param p = new Param();
			p.l2 = this.l2;
			LinearModel b = this.train(p, trainFeatures, trainLabels);
			this.mdl = b;
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException(e.getMessage());
		}
	}

	@Override
	public float[] predict(String[] smiles, int[] columns) {
		float[][] features = features(smiles, columns);
		float[] predictions = ArUtls.mult(1000, ArUtls.toFloatArray(mdl.predict(ArUtls.toDataFrame(features))));
		return predictions;
	}

	@Override
	public void save(String directory) throws IOException {
		Files.createDirectories(Paths.get(directory));
		FileWriter fw = new FileWriter(new File(directory, "ModelType.txt"));
		fw.write(this.modelType());
		fw.close();
		fw = new FileWriter(new File(directory, "info.txt"));
		fw.write(l2 + "");
		fw.close();
		fw = new FileWriter(new File(directory, "info1.txt"));
		fw.write(this.fullModelInfo());
		fw.close();
		File f = new File(directory, "model.xml");
		fw = new FileWriter(f);
		XStream xstream = new XStream(new StaxDriver());
		xstream.toXML(mdl, fw);
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
		this.l2 = Float.parseFloat(s);

		File f = new File(directory, "model.xml");
		XStream xstream = new XStream();
		xstream.allowTypes(new String[] { "smile.regression.LinearModel", "smile.data.formula.Variable",
				"smile.data.type.StructField", "smile.data.type.DoubleType" });
		mdl = (LinearModel) xstream.fromXML(f);
	}

	@Override
	public String modelType() {
		return "RIDGE";
	}

	@Override
	public String fullModelInfo() {
		return "RIDGE " + " l2 " + l2;
	}

	@Override
	public ModelRI createSimilar() {
		RidgeRI result = (RidgeRI) QSRRModelRI.getModel(this.modelType(), this.getGenCol(), this.getGen(),
				this.isHyperparamsTune(), this.getHyperparamsTuneFile(), this.getHyperparamsTuneAttempts());
		result.l2=this.l2;
		return result;
	}

}
