package ru.ac.phyche.ribyclusters.clusters;

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
import smile.clustering.DBSCAN;
import smile.projection.PCA;

public class DBSCANClustering extends Clustering {
	private DBSCAN<double[]> cl = null;
	private int nPCA = 10;
	private boolean pcaRequired = false;
	private PCA pca = null;
	private int minPts = 100;
	private float radius = 0.1f;

	@Override
	public int[] train(float[][] features) {
		double[][] featuresDouble = ArUtls.toDoubleArray2d(features);
		if (pcaRequired) {
			nPCA = Math.min(Math.min(nPCA, features.length), features[0].length);
			pca = PCA.fit(featuresDouble);
			pca.setProjection(nPCA);
			featuresDouble = pca.project(featuresDouble);
		}
		cl = DBSCAN.fit(featuresDouble, minPts, radius);
		int[] r = new int[features.length];
		this.setClustersNum(cl.k + 1);

		for (int i = 0; i < r.length; i++) {
			r[i] = cl.predict(featuresDouble[i]);
			if (r[i] > this.getClustersNum()) {
				r[i] = this.getClustersNum() - 1;
			}
		}
		return r;
	}

	@Override
	public void save(String directoryName) throws IOException {
		Files.createDirectories(Paths.get(directoryName));
		FileWriter fw = new FileWriter(new File(directoryName, "ModelType.txt"));
		fw.write(this.modelType());
		fw.close();
		fw = new FileWriter(new File(directoryName, "info.txt"));
		fw.write(this.getClustersNum() + " " + (this.pcaRequired ? "1" : "0"));
		fw.close();
		fw = new FileWriter(new File(directoryName, "dbscan.xml"));
		XStream xstream = new XStream(new StaxDriver());
		xstream.toXML(cl, fw);
		fw.close();
		if (pcaRequired) {
			fw = new FileWriter(new File(directoryName, "pca.xml"));
			xstream = new XStream(new StaxDriver());
			xstream.toXML(pca, fw);
			fw.close();
		}
	}

	@SuppressWarnings("unchecked")
	@Override
	public void load(String directoryName) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(directoryName, "ModelType.txt")));
		String s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		if (!s.trim().equals(this.modelType())) {
			throw new RuntimeException("Wrong model type");
		}
		br = new BufferedReader(new FileReader(new File(directoryName, "info.txt")));
		s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		String[] split = s.trim().split("\\s+");
		this.setClustersNum(Integer.parseInt(split[0]));
		this.pcaRequired = Float.parseFloat(split[1]) >= 0.5;
		XStream xstream = new XStream();
		xstream.allowTypes(new String[] { "smile.clustering.DBSCAN", "smile.neighbor.KDTree" });
		cl = (DBSCAN<double[]>) xstream.fromXML(new File(directoryName, "dbscan.xml"));
		xstream = new XStream();
		if (pcaRequired) {
			xstream.allowTypes(
					new String[] { "smile.projection.PCA", "smile.math.matrix.Matrix", "smile.math.matrix.Matrix$1" });
			pca = (PCA) xstream.fromXML(new File(directoryName, "pca.xml"));
			this.nPCA = pca.getProjection().nrows();
		}
	}

	@Override
	public int[] predict(float[][] features) {
		double[][] featuresDouble = ArUtls.toDoubleArray2d(features);
		if (pcaRequired) {
			pca.setProjection(nPCA);
			double[][] pcaD = pca.project(featuresDouble);
			if (pcaD[0].length != nPCA) {
				throw new RuntimeException("Wrong matrix dimension after PCA");
			}
			if (pcaD.length != features.length) {
				throw new RuntimeException("Wrong matrix dimension after PCA");
			}
			featuresDouble = pcaD;
		}
		if (featuresDouble[0].length != nPCA) {
			throw new RuntimeException("Wrong matrix dimension after PCA");
		}
		if (featuresDouble.length != features.length) {
			throw new RuntimeException("Wrong matrix dimension after PCA");
		}
		int[] r = new int[features.length];
		for (int i = 0; i < r.length; i++) {
			r[i] = cl.predict(featuresDouble[i]);
			if (r[i] > this.getClustersNum()) {
				r[i] = this.getClustersNum() - 1;
			}
		}
		return r;
	}

	@Override
	public void init(float[] parameters) {
		if (parameters.length != 4) {
			throw new RuntimeException("Wrong number of parameters. KMeansIterative has 4 parameters.");
		}
		this.nPCA = Math.round(parameters[0]);
		this.pcaRequired = parameters[1] >= 0.5 && this.nPCA > 1;
		this.minPts = Math.round(parameters[2]);
		this.radius = parameters[3];
	}

	@Override
	public void init() {
		init(new float[] { 1, 10, 100, 0.1f });
	}

	@Override
	public String modelType() {
		return "DBSCANClustering";
	}

	@Override
	public String paramsNames() {
		return "nPCA pcaRequired minPts radius";
	}
}