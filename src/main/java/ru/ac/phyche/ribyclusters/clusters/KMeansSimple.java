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
import smile.clustering.KMeans;

public class KMeansSimple extends Clustering {

	private KMeans km = null;

	@Override
	public int[] train(float[][] features) {
		double[][] f = ArUtls.toDoubleArray2d(features);
		km = KMeans.fit(f, this.getClustersNum());
		int[] r = new int[features.length];
		for (int i = 0; i < r.length; i++) {
			r[i] = km.predict(f[i]);
		}
		return r;
	}

	@Override
	public void save(String directoryName) throws IOException {
		Files.createDirectories(Paths.get(directoryName));
		FileWriter fw = new FileWriter(new File(directoryName, "ModelType.txt"));
		fw.write(this.modelType());
		fw.close();
		fw = new FileWriter(new File(directoryName, "k.txt"));
		fw.write(this.getClustersNum()+"");
		fw.close();
		fw = new FileWriter(new File(directoryName, "kmeans.xml"));
		XStream xstream = new XStream(new StaxDriver());
		xstream.toXML(km, fw);
		fw.close();
	}

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
		br = new BufferedReader(new FileReader(new File(directoryName, "k.txt")));
		s = br.readLine();
		while (s.trim().equals("")) {
			s = br.readLine();
		}
		br.close();
		this.setClustersNum(Integer.parseInt(s.trim()));
		XStream xstream = new XStream();
		xstream.allowTypes(new String[] {});
		km = (KMeans) xstream.fromXML(new File(directoryName, "kmeans.xml"));
	}

	@Override
	public int[] predict(float[][] features) {
		double[][] f = ArUtls.toDoubleArray2d(features);
		int[] r = new int[features.length];
		for (int i = 0; i < r.length; i++) {
			r[i] = km.predict(f[i]);
		}
		return r;
	}

	@Override
	public void init(float[] parameters) {
		this.setClustersNum(Math.round(parameters[0]));
		if (parameters.length != 0) {
			throw new RuntimeException("Wrong number of parameters. KMeansSimple has 1 parameter.");
		}
	}

	@Override
	public void init() {
		init(new float[] {10});
	}

	@Override
	public String modelType() {
		return "KMeansSimple";
	}

	@Override
	public String paramsNames() {
		return "k";
	}

}
