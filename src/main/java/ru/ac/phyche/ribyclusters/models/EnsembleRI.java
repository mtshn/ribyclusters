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
import smile.regression.OLS;

public class EnsembleRI extends ModelRI {

	ModelRI models[];
	LinearModel ensemble;
	ColumnFeatures[] genCol;
	FeaturesGenerator[] gen;
	boolean trainOnlySecondLevel = false;

	public EnsembleRI(ColumnFeatures[] genCol, FeaturesGenerator[] gen, ModelRI models[]) throws IOException {
		this.genCol = genCol;
		this.gen = gen;
		this.models = models;
	}

	public EnsembleRI(ColumnFeatures[] genCol, FeaturesGenerator[] gen, ModelRI models[],
			boolean trainOnlySecondLevel) throws IOException {
		this.genCol = genCol;
		this.gen = gen;
		this.models = models;
		this.trainOnlySecondLevel = trainOnlySecondLevel;
	}

	public EnsembleRI(ColumnFeatures[] genCol, FeaturesGenerator[] gen) throws IOException {
		this.genCol = genCol;
		this.gen = gen;
		this.models = new ModelRI[genCol.length];
		for (int i = 0; i < genCol.length; i++) {
			models[i] = new RidgeRI(genCol[i], gen[i]);
		}
	}

	public EnsembleRI(String[] modelsTypes, ColumnFeatures[] genCol, FeaturesGenerator[] gen,
			int[] hyperparamsTuneAttempts, String hyperparamsTuneDir) throws IOException {
		this.genCol = genCol;
		this.gen = gen;
		if (hyperparamsTuneDir != null) {
			Files.createDirectories(Paths.get(hyperparamsTuneDir));
		}
		int n = modelsTypes.length;
		if ((genCol.length != n) || (gen.length != n) || (hyperparamsTuneAttempts.length != n)) {
			throw new RuntimeException("Wrong array length");
		}
		models = new QSRRModelRI[n];
		for (int i = 0; i < n; i++) {
			File f = null;
			if (hyperparamsTuneDir != null) {
				f = new File(hyperparamsTuneDir, "model" + i + ".txt");
			}
			models[i] = QSRRModelRI.getModel(modelsTypes[i], genCol[i], gen[i], hyperparamsTuneAttempts[i] > 0,
					f == null ? null : f.getAbsolutePath(), hyperparamsTuneAttempts[i]);
		}
	}

	@Override
	public float[] predict(String[] smiles, int[] columns) {
		float[][] predictionsT = new float[models.length][smiles.length];
		for (int i = 0; i < models.length; i++) {
			predictionsT[i] = ArUtls.mult(0.001f, models[i].predict(smiles, columns));
		}
		float[][] predictions = ArUtls.transpose(predictionsT);
		float[] result = ArUtls.mult(1000, ArUtls.toFloatArray(ensemble.predict(ArUtls.toDataFrame(predictions))));
		return result;
	}

	@Override
	public void train(ChemDataset trainSet, ChemDataset validationSet) {
		if (!trainOnlySecondLevel) {
			for (int i = 0; i < models.length; i++) {
				models[i].train(trainSet, validationSet);
			}
		}
		float[][] predictionsT = new float[models.length][validationSet.size()];
		for (int i = 0; i < models.length; i++) {
			predictionsT[i] = ArUtls.mult(0.001f, models[i].predict(validationSet));
		}
		float[][] predictions = ArUtls.transpose(predictionsT);
		DataFrame dataFrame = ArUtls.toDataFrame(predictions, ArUtls.mult(0.001F, validationSet.allRetentions()));
		ensemble = OLS.fit(Formula.lhs("label"), dataFrame);
	}

	@Override
	public void save(String directory) throws IOException {
		for (int i = 0; i < models.length; i++) {
			File f = null;
			f = new File(directory, "model" + i);
			models[i].save(f.getAbsolutePath());
		}
		FileWriter fw = new FileWriter(new File(directory, "ModelType.txt"));
		fw.write(this.modelType());
		fw.close();
		fw = new FileWriter(new File(directory, "info.txt"));
		fw.write(models.length + "");
		fw.close();
		fw = new FileWriter(new File(directory, "info1.txt"));
		fw.write(this.fullModelInfo());
		fw.close();
		File f = new File(directory, "model.xml");
		fw = new FileWriter(f);
		XStream xstream = new XStream(new StaxDriver());
		xstream.toXML(ensemble, fw);
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
		int n = Integer.parseInt(s);
		models = new QSRRModelRI[n];
		File f = new File(directory, "model.xml");
		XStream xstream = new XStream();
		xstream.allowTypes(new String[] { "smile.regression.LinearModel", "smile.data.formula.Variable",
				"smile.data.type.StructField", "smile.data.type.DoubleType", "smile.math.matrix.Matrix" });
		ensemble = (LinearModel) xstream.fromXML(f);
		if (n != genCol.length) {
			throw new RuntimeException("Wrong array length");
		}
		for (int i = 0; i < models.length; i++) {
			f = null;
			f = new File(directory, "model" + i);
			models[i] = QSRRModelRI.loadModel(f.getAbsolutePath(), this.genCol[i], this.gen[i]);
		}
	}

	@Override
	public String modelType() {
		return "Ensemble";
	}

	@Override
	public String fullModelInfo() {
		String s = "EnsembleRI " + models.length + " ";
		for (int i = 0; i < models.length; i++) {
			s = s + models[i].fullModelInfo() + " ";
		}
		return s;
	}

	@Override
	public ModelRI createSimilar() {
		String[] modelsTypes = new String[this.models.length];
		int[] hyperparamsTuneAttempts = new int[this.models.length];
		ColumnFeatures[] genCol = new ColumnFeatures[this.models.length];
		FeaturesGenerator[] gen = new FeaturesGenerator[this.models.length];
		for (int i = 0; i < models.length; i++) {
			modelsTypes[i] = this.models[i].modelType();
			QSRRModelRI m =(QSRRModelRI)this.models[i];
			hyperparamsTuneAttempts[i] = m.getHyperparamsTuneAttempts();
			genCol[i] = m.getGenCol();
			gen[i] = m.getGen();
		}

		EnsembleRI result;
		try {
			QSRRModelRI m =(QSRRModelRI)this.models[0];
			File f = new File(m.getHyperparamsTuneFile());
			result = new EnsembleRI(modelsTypes, genCol, gen, hyperparamsTuneAttempts, f.getParent());
		} catch (IOException e) {
			e.printStackTrace();
			throw new RuntimeException(e.getMessage());
		}
		return result;
	}

	@Override
	public void setTuningOutFileOrDir(String filename) {
		if (filename != null) {
			for (int i = 0; i < models.length; i++) {
				File f = null;
				if (filename != null) {
					f = new File(filename, "model" + i + ".txt");
				}
				QSRRModelRI m =(QSRRModelRI)models[i];
				m.setHyperparamsTuneFile(f.getAbsolutePath());
			}
		} else {
			for (int i = 0; i < models.length; i++) {
				QSRRModelRI m =(QSRRModelRI)models[i];
				m.setHyperparamsTuneFile(null);
			}
		}
	}

}
