package ru.ac.phyche.ribyclusters;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import ru.ac.phyche.ribyclusters.clusters.Clustering;
import ru.ac.phyche.ribyclusters.featuregenerators.FeaturesGenerator;
import ru.ac.phyche.ribyclusters.models.EnsembleRI;
import ru.ac.phyche.ribyclusters.models.ModelRI;
import ru.ac.phyche.ribyclusters.models.QSRRModelRI;

public class RIByClusters extends ModelRI {

	private Clustering cl;
	private FeaturesGenerator genForClusters;
	private ModelRI[] models;
	private ModelRI exampleModel;
	private String outputTrainingDir;

	public static RIByClusters getInstance(Clustering cl, FeaturesGenerator genForClusters, ModelRI exampleModel,
			String outputTrainingDir) {
		RIByClusters result = new RIByClusters();
		result.outputTrainingDir = outputTrainingDir;
		result.cl = cl;
		result.genForClusters = genForClusters;
		result.exampleModel = exampleModel;
		return result;
	}

	@Override
	public float[] predict(String[] smiles, int[] columns) {
		genForClusters.precompute(smiles);
		int[] ns = cl.predict(genForClusters.features(smiles));
		float[] result = new float[smiles.length];
		for (int i = 0; i < ns.length; i++) {
			result[i] = models[ns[i]].predict(smiles[i], columns[i]);
		}
		return result;
	}

	@Override
	public void train(ChemDataset trainSet, ChemDataset validationSet) {
		cl.train(trainSet.compounds().toArray(new String[trainSet.compounds().size()]), genForClusters);
		this.models = new ModelRI[cl.getClustersNum()];
		for (int i = 0; i < models.length; i++) {
			models[i] = exampleModel.createSimilar();
		}
		ChemDataset[] trainSubsets = cl.predictAndSplit(trainSet, genForClusters);
		ChemDataset[] valSubsets = cl.predictAndSplit(validationSet, genForClusters);
		FileWriter fw = null;
		if (outputTrainingDir != null) {
			try {
				Files.createDirectories(Paths.get(outputTrainingDir));
				File f1 = new File(outputTrainingDir, "trainingClustered");
				File f2 = new File(outputTrainingDir, "validationClustered");
				Files.createDirectories(Paths.get(f1.getAbsolutePath()));
				Files.createDirectories(Paths.get(f2.getAbsolutePath()));
				for (int i = 0; i < cl.getClustersNum(); i++) {
					trainSubsets[i].saveToFile((new File(f1.getAbsolutePath(), i + ".ri")).getAbsolutePath());
					valSubsets[i].saveToFile((new File(f2.getAbsolutePath(), i + ".ri")).getAbsolutePath());
				}
				File f3 = new File(outputTrainingDir, "modelsTuning");
				Files.createDirectories(Paths.get(f3.getAbsolutePath()));
				for (int i = 0; i < models.length; i++) {
					File f4 = new File(f3.getAbsolutePath(), "tuning" + i);
					models[i].setTuningOutFileOrDir(f4.getAbsolutePath());
				}
				fw = new FileWriter(new File(outputTrainingDir, "clusterTrain.txt"));
			} catch (IOException e) {
				e.printStackTrace();
				throw new RuntimeException(e.getMessage());
			}
		}

		for (int i = 0; i < models.length; i++) {
			models[i].train(trainSubsets[i], valSubsets[i]);
			if (outputTrainingDir != null) {
				try {
					fw.write("cluster " + i + " ");
					fw.write("clusterSize " + trainSubsets[i].size() + " " + valSubsets[i].size() + " ");
					fw.write(models[i].validate(valSubsets[i], false) + " ");
					fw.write(models[i].fullModelInfo() + " ");
					fw.write(trainSubsets[i].countIdenticalByInchi(valSubsets[i]) + " ");
					fw.write(valSubsets[i].countIdenticalByCanonicalSmiles(trainSubsets[i]) + "\n");
					fw.flush();
				} catch (Exception e) {
					try {
						fw.close();
					} catch (IOException e1) {
						e1.printStackTrace();
					}
					e.printStackTrace();
					throw new RuntimeException(e.getMessage());
				}
			}
		}
		if (fw != null) {
			try {
				fw.close();
			} catch (IOException e) {
				e.printStackTrace();
				throw new RuntimeException(e.getMessage());
			}
		}
	}

	@Override
	public void save(String directory) throws IOException {
		for (int i = 0; i < models.length; i++) {
			File f = null;
			f = new File(directory, "model_for_cluster_" + i);
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
		cl.save(new File(directory, "clustering_model").getAbsolutePath());
	}

	@Override
	public void load(String directory) throws IOException {
		cl.load(new File(directory, "clustering_model").getAbsolutePath());
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
		if (n != cl.getClustersNum()) {
			throw new RuntimeException("Wrong number of models and/or clusters");
		}
		this.models = new ModelRI[cl.getClustersNum()];
		for (int i = 0; i < n; i++) {
			models[i] = exampleModel.createSimilar();
			File f = null;
			f = new File(directory, "model_for_cluster_" + i);
			models[i].load(f.getAbsolutePath());
		}
	}

	public static RIByClusters load(String directory, FeaturesGenerator genForClusters, FeaturesGenerator genForModels,
			ColumnFeatures forModels) throws IOException {
		RIByClusters result = new RIByClusters();
		result.cl = Clustering.loadModel(new File(directory, "clustering_model").getAbsolutePath());
		BufferedReader br = new BufferedReader(new FileReader(new File(directory, "ModelType.txt")));
		String s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		if (!s.trim().equals(result.modelType())) {
			throw new RuntimeException("Wrong model type");
		}
		br = new BufferedReader(new FileReader(new File(directory, "info.txt")));
		s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		int n = Integer.parseInt(s);
		result.models = new QSRRModelRI[n];
		if (n != result.cl.getClustersNum()) {
			throw new RuntimeException("Wrong number of models and/or clusters");
		}
		for (int i = 0; i < n; i++) {
			File f = null;
			f = new File(directory, "model_for_cluster_" + i);
			result.models[i] = QSRRModelRI.loadModel(f.getAbsolutePath(), forModels, genForModels);
		}
		result.genForClusters = genForClusters;
		return result;
	}

	public static RIByClusters loadEnsembles(String directory, FeaturesGenerator genForClusters,
			FeaturesGenerator[] genForModels, ColumnFeatures[] forModels) throws IOException {
		RIByClusters result = new RIByClusters();
		result.cl = Clustering.loadModel(new File(directory, "clustering_model").getAbsolutePath());
		BufferedReader br = new BufferedReader(new FileReader(new File(directory, "ModelType.txt")));
		String s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		if (!s.trim().equals(result.modelType())) {
			throw new RuntimeException("Wrong model type");
		}
		br = new BufferedReader(new FileReader(new File(directory, "info.txt")));
		s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		int n = Integer.parseInt(s);
		result.models = new QSRRModelRI[n];
		if (n != result.cl.getClustersNum()) {
			throw new RuntimeException("Wrong number of models and/or clusters");
		}
		for (int i = 0; i < n; i++) {
			File f = null;
			f = new File(directory, "model_for_cluster_" + i);
			result.models[i] = new EnsembleRI(forModels, genForModels);
			result.models[i].load(f.getAbsolutePath());
		}
		result.genForClusters = genForClusters;
		return result;
	}

	@Override
	public String modelType() {
		return "RIByClusters";
	}

	@Override
	public String fullModelInfo() {
		String s = modelType() + " clusters_num: ";
		s = s + cl.getClustersNum() + " " + cl.modelType() + " ";
		for (int i = 0; i < cl.getClustersNum(); i++) {
			s = s + "model " + i + " " + models[i].fullModelInfo() + " ";
		}
		return s;
	}

	private RIByClusters() {

	}

	@Override
	public ModelRI createSimilar() {
		return RIByClusters.getInstance(cl, genForClusters, models[0], outputTrainingDir);
	}

	@Override
	public void setTuningOutFileOrDir(String filename) {
		this.outputTrainingDir = filename;
	}

}
